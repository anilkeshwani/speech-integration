import random
from enum import Enum
from functools import partial
from itertools import groupby, zip_longest
from tkinter.tix import TEXT
from typing import Any, Callable, Mapping

import numpy as np
from datasets import load_dataset
from datasets.dataset_dict import IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from sardalign.constants import (
    ALIGNMENT_END_TIME_KEY,
    ALIGNMENT_START_TIME_KEY,
    HUBERT_DOWNSAMPLING_RATIO,
    MODALITY_TOKEN_SPEECH,
    MODALITY_TOKEN_TEXT,
    SAMPLING_FREQ,
    SPEECH_TOKENS_KEY,
    TOKENIZED_KEY,
)
from sardalign.utils import dsu2pua
from sardalign.utils.align import times_to_hubert_idxs as times_to_dsu_idxs
from torch import Type
from torch.utils.data import Dataset
from torchtune.data._utils import truncate
from torchtune.models.llama3 import Llama3Tokenizer

from ssi.constants import SEED


class CompletionSequenceType(Enum):
    INTERLEAVED = "interleaved"  # interleaved text-speech sequences
    CONCATENATED_TXT_DSU = "concatenated_txt_dsu"  # concatenated text and DSU sequences
    CONCATENATED_DSU_TXT = "concatenated_dsu_txt"  # concatenated DSU and text sequences
    DSU_ONLY = "dsu_only"  # DSU-only sequences
    TEXT_ONLY = "text_only"  # text-only sequences (i.e. regular text completion)
    ALTERNATING = "alternating"  # alternating between text-only and DSU-only sequences


# Module-level pseudo-random number generator
PRNG = np.random.default_rng(SEED)

# Constants from sardalign (speech-text-alignment) in scripts/interleave.py
MEAN_MLS_SEQ_LEN: float = 39.43  # mean sequence length (in tokens) of the MLS stratified sample; 25% of en trainset
BINOM_PROB: float = 0.1  # fraction of sequence to make up subspans


class TextCompletionDataset(Dataset):
    """
    Freeform dataset for any unstructured text corpus. Quickly load any dataset
    from Hugging Face or local disk and tokenize it for your model.

    Args:
        tokenizer (Llama3Tokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
            for more details.
        column (str): name of column in the sample that contains the text data. This is typically required
            for Hugging Face datasets or tabular data. For local datasets with a single column
            (e.g. unstructured txt files), use the default "text" which is used by Hugging Face datasets
            when loaded into memory. Default is "text".
        add_eos (bool): Whether to add an EOS token to the end of the sequence. Default is True.
        filter_fn (Callable | None): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        **load_dataset_kwargs (dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details.
    """

    def __init__(
        self,
        tokenizer: Llama3Tokenizer,
        source: str,
        sequence_type: CompletionSequenceType,
        deduplicate: bool,
        use_modality_tokens: bool,
        add_eos: bool = True,
        tokenized_key: str | None = None,
        alignment_start_time_key: str | None = None,
        alignment_end_time_key: str | None = None,
        speech_tokens_key: str | None = None,
        filter_fn: Callable | None = None,
        **load_dataset_kwargs: dict[str, Any],  # TODO make HF dataset-specific args explicit
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        self.add_eos = add_eos

        # dataset columns
        if tokenized_key is None:
            tokenized_key = TOKENIZED_KEY
        if alignment_start_time_key is None:
            alignment_start_time_key = ALIGNMENT_START_TIME_KEY
        if alignment_end_time_key is None:
            alignment_end_time_key = ALIGNMENT_END_TIME_KEY
        if speech_tokens_key is None:
            speech_tokens_key = SPEECH_TOKENS_KEY

        self.sequence_type = sequence_type
        match self.sequence_type:
            case CompletionSequenceType.INTERLEAVED:
                # TODO after all args are explicit, pass in specific kwargs and make partial
                self.prompt_fn = interleave
            case CompletionSequenceType.CONCATENATED_TXT_DSU:
                self.prompt_fn = concat_txt_dsu
            case CompletionSequenceType.CONCATENATED_DSU_TXT:
                self.prompt_fn = concat_dsu_txt
            case _:
                raise ValueError(f"Unsupported sequence type: {self.sequence_type}")

        self.deduplicate = deduplicate
        self.use_modality_tokens = use_modality_tokens

        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> dict[str, list[int]]:
        # Construct the prompt
        prompt = self.prompt_fn(
            sample=sample,
            deduplicate=self.deduplicate,
            use_modality_tokens=self.use_modality_tokens,
        )

        print(f"Prompt: \n{prompt}")

        # Tokenize
        tokens = self._tokenizer.encode(text=prompt, add_bos=True, add_eos=self.add_eos)

        # Truncate if needed, but don't coerce EOS id
        if self._tokenizer.max_seq_len is not None:
            tokens = truncate(tokens, self._tokenizer.max_seq_len - 1)

        # No need to offset labels by 1 - happens in the recipe
        labels = tokens.copy()

        return {"tokens": tokens, "labels": labels}


def get_span_idxs_binomial(n: int, p: float, seq_len: int, seed: int = SEED) -> list[int]:
    subspan_idxs = np.maximum(PRNG.binomial(n, p, size=seq_len), 1).cumsum()
    return [0] + subspan_idxs[subspan_idxs < seq_len].tolist() + [seq_len]


def interleave(
    sample: dict[str, Any],
    deduplicate: bool,
    use_modality_tokens: bool,
    mean_seq_len: float = MEAN_MLS_SEQ_LEN,
    binom_prob: float = BINOM_PROB,
    seed: int = SEED,
) -> str:
    start_with_text = PRNG.choice([True, False], p=[0.5, 0.5])
    tokens = sample[TOKENIZED_KEY]
    align_t_starts = sample[ALIGNMENT_START_TIME_KEY]
    align_t_ends = sample[ALIGNMENT_END_TIME_KEY]
    speech_tokens: list[int] = sample[SPEECH_TOKENS_KEY]
    span_idxs = get_span_idxs_binomial(int(mean_seq_len), binom_prob, len(tokens), seed)
    # idxs: list of 2-tuples of start and end indices of subspans e.g. [(0, 4), (11, 16), (21, 25), (28, 31)]
    idxs1, idxs2 = zip(span_idxs[:-1:2], span_idxs[1::2]), zip(span_idxs[1:-1:2], span_idxs[2::2])
    text_idxs, dsu_idxs = (idxs1, idxs2) if start_with_text else (idxs2, idxs1)
    text_spans: list[str] = [" ".join(tokens[start_idx:end_idx]) for start_idx, end_idx in text_idxs]
    dsu_spans: list[str] = []
    for start_idx, end_idx in dsu_idxs:
        start_idx_hu, end_idx_hu = times_to_dsu_idxs(
            (align_t_starts[start_idx], align_t_ends[end_idx - 1]),
            SAMPLING_FREQ,  # TODO parameterize BUT 16kHz should be the default/only supported SR
            HUBERT_DOWNSAMPLING_RATIO,  # TODO parameterize for SpeechTokenizer, Mimi, etc.
        )
        sp_tkns_spn = speech_tokens[start_idx_hu:end_idx_hu]
        if deduplicate:
            sp_tkns_spn = [k for k, g in groupby(sp_tkns_spn)]
        dsu_spans.append("".join([dsu2pua(sp_tkn) for sp_tkn in sp_tkns_spn]))

    if use_modality_tokens:
        text_spans = [" ".join((MODALITY_TOKEN_TEXT, text_span)) for text_span in text_spans]
        dsu_spans = [" ".join((MODALITY_TOKEN_SPEECH, dsu_span)) for dsu_span in dsu_spans]

    mm_spans = (text_spans, dsu_spans) if start_with_text else (dsu_spans, text_spans)
    interleaved_segment = " ".join([span for spans in zip_longest(*mm_spans) for span in spans if span is not None])
    return interleaved_segment


def _concatenate_speech_text(
    sample: dict[str, Any],
    deduplicate: bool,
    use_modality_tokens: bool,
    start_with_text: bool,
) -> str:
    speech_tokens: list[int] = sample[SPEECH_TOKENS_KEY]
    if deduplicate:
        speech_tokens = [k for k, g in groupby(speech_tokens)]
    text: str = " ".join(sample[TOKENIZED_KEY])
    dsus_str: str = "".join([dsu2pua(sp_tkn) for sp_tkn in speech_tokens])
    if use_modality_tokens:
        text = " ".join((MODALITY_TOKEN_TEXT, text))
        dsus_str = " ".join((MODALITY_TOKEN_SPEECH, dsus_str))  # NOTE includes a leading space
    return " ".join((text, dsus_str) if start_with_text else (dsus_str, text))


concat_txt_dsu = partial(_concatenate_speech_text, start_with_text=True)
concat_dsu_txt = partial(_concatenate_speech_text, start_with_text=False)
