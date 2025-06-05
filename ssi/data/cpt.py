import logging
from enum import Enum
from functools import partial
from itertools import groupby, zip_longest
from typing import Any, Callable, Mapping

import numpy as np
from datasets import load_dataset
from sardalign.constants import (
    ALIGNMENT_END_TIME_KEY,
    ALIGNMENT_START_TIME_KEY,
    MODALITY_TOKEN_SPEECH,
    MODALITY_TOKEN_TEXT,
    SPEECH_TOKENS_KEY,
    TOKENIZED_KEY,
)
from sardalign.utils import dsu2pua
from sardalign.utils.align import times_to_hubert_idxs as times_to_dsu_idxs
from torch.utils.data import Dataset
from torchtune.data._utils import truncate
from torchtune.models.llama3 import Llama3Tokenizer

from ssi.constants import SEED


LOGGER = logging.getLogger(__name__)


class CompletionSequenceType(Enum):
    INTERLEAVED = "interleaved"  # interleaved text-speech sequences
    CONCATENATED_TXT_DSU = "concatenated_txt_dsu"  # concatenated text and DSU sequences
    CONCATENATED_DSU_TXT = "concatenated_dsu_txt"  # concatenated DSU and text sequences

    # Not implemented yet
    DSU_ONLY = "dsu_only"  # DSU-only sequences
    TEXT_ONLY = "text_only"  # text-only sequences (i.e. regular text completion)
    ALTERNATING = "alternating"  # alternating between text-only and DSU-only sequences


# Module-level pseudo-random number generator
PRNG = np.random.default_rng(SEED)


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
        add_eos (bool): Whether to add an EOS token to the end of the sequence. Default is True.
        filter_fn (Callable | None): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): split of the dataset to load. Default is "train". See Hugging Face's
            `load_dataset <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
    """

    def __init__(
        self,
        tokenizer: Llama3Tokenizer,
        source: str,
        split: str,
        sequence_type: str,
        deduplicate: bool,
        use_modality_tokens: bool,
        add_eos: bool = True,
        tokenized_key: str | None = None,
        alignment_start_time_key: str | None = None,
        alignment_end_time_key: str | None = None,
        speech_tokens_key: str | None = None,
        filter_fn: Callable | None = None,
        interleave_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(source, split=split)
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

        self.sequence_type = CompletionSequenceType(sequence_type)
        match self.sequence_type:
            case CompletionSequenceType.INTERLEAVED:
                if not interleave_kwargs:
                    raise ValueError("interleave_kwargs must be provided for interleaved sequence type")
                self.prompt_fn = partial(interleave, **interleave_kwargs)
            case CompletionSequenceType.CONCATENATED_TXT_DSU:
                self.prompt_fn = partial(concatenate_speech_text, start_with_text=True)
            case CompletionSequenceType.CONCATENATED_DSU_TXT:
                self.prompt_fn = partial(concatenate_speech_text, start_with_text=False)
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

        # Tokenize
        tokens = self._tokenizer.encode(text=prompt, add_bos=True, add_eos=self.add_eos)

        LOGGER.debug(f"Prompt Length: {len(prompt)}")
        LOGGER.debug(f"Tokens Length: {len(tokens)}")
        LOGGER.debug(f"Prompt: \n{prompt}")
        LOGGER.debug(f"Tokens: \n{tokens}")

        # Truncate if needed, but don't coerce EOS id
        if self._tokenizer.max_seq_len is not None:
            # TODO -1 is odd and does not match the tokenize_messages method NOTE this is original torchtune code
            tokens = truncate(tokens, self._tokenizer.max_seq_len - 1)

        # No need to offset labels by 1 - happens in the recipe
        labels = tokens.copy()

        return {"tokens": tokens, "labels": labels}


def get_span_idxs_binomial(n: int, p: float, seq_len: int) -> list[int]:
    subspan_idxs = np.maximum(PRNG.binomial(n, p, size=seq_len), 1).cumsum()  # NOTE sample lower bounded to 1
    return [0] + subspan_idxs[subspan_idxs < seq_len].tolist() + [seq_len]


def interleave(
    sample: dict[str, Any],
    deduplicate: bool,
    use_modality_tokens: bool,
    *,
    sampling_rate: int,
    downsampling_ratio: int,
    mean_seq_len_tokens: float,
    binom_prob: float,
) -> str:
    start_with_text = PRNG.choice([True, False], p=[0.5, 0.5])
    tokens = sample[TOKENIZED_KEY]
    align_t_starts = sample[ALIGNMENT_START_TIME_KEY]
    align_t_ends = sample[ALIGNMENT_END_TIME_KEY]
    speech_tokens: list[int] = sample[SPEECH_TOKENS_KEY]
    span_idxs = get_span_idxs_binomial(int(mean_seq_len_tokens), binom_prob, len(tokens))
    # idxs: list of 2-tuples of start and end indices of subspans e.g. [(0, 4), (11, 16), (21, 25), (28, 31)]
    idxs1, idxs2 = zip(span_idxs[:-1:2], span_idxs[1::2]), zip(span_idxs[1:-1:2], span_idxs[2::2])
    text_idxs, dsu_idxs = (idxs1, idxs2) if start_with_text else (idxs2, idxs1)
    text_spans: list[str] = [" ".join(tokens[start_idx:end_idx]) for start_idx, end_idx in text_idxs]
    dsu_spans: list[str] = []
    for start_idx, end_idx in dsu_idxs:
        start_idx_hu, end_idx_hu = times_to_dsu_idxs(
            (align_t_starts[start_idx], align_t_ends[end_idx - 1]),
            sampling_rate,
            downsampling_ratio,
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


def concatenate_speech_text(
    sample: dict[str, Any],
    deduplicate: bool,
    use_modality_tokens: bool,
    *,
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
