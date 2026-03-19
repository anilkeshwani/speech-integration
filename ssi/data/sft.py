from collections.abc import Callable, Mapping
from itertools import groupby
import logging
from pathlib import Path
from typing import Any

import datasets
from datasets import load_dataset
import numpy as np
from sardalign.constants import MODALITY_TOKEN_SPEECH, MODALITY_TOKEN_TEXT
from sardalign.utils import dsu2pua
from torch.utils.data import Dataset
from torchtune.data import Message
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.data._messages import validate_messages
from torchtune.data._utils import load_image
from torchtune.models.llama3 import Llama3Tokenizer

from ssi.constants import RESERVED_BATCH_KEYS


LOGGER = logging.getLogger(__name__)


class SFTDataset(Dataset):
    """
    Supervised Finetuning Dataset class based on torchtune.datasets._sft.SFTDataset (torchtune v0.5.0)

    Modifications:
    - The ``message_transform``  parameter is constrained to be the InputOutputToMessages class,
      defined in the same module. ``InputOutputToMessages.__call__``  supports a ``inference`` parameter
    - The model_transform parameter is replaced with the model_tokenizer parameter, which is constrained
      to be the Llama3Tokenizer class

    Creates SFT dataset from Hugging Face Hub, local files, or remote files.
    Supports instruct, chat, tool or multimodal data for fine-tuning.
    This class loads the data from source and applies these pre-processing steps:

    1. Dataset-specific transform: typically unique to each dataset and extracts
       the necessary columns into torchtune's :class:`~torchtune.data.Message` format,
       a standardized API for all model tokenizers.
    2. Model-specific transform or tokenization with optional prompt template

    All datasets are formatted into a list of :class:`~torchtune.data.Message`
    because for fine-tuning, datasets can be considered as "conversations" with the model,
    or AI assistant.

    All text content is standardized to messages in a conversation assigned to a role:
    - ``"system"`` messages contain the system prompt
    - ``"user"`` messages contain the input prompt into the model
    - ``"assistant"`` messages are the response of the model and what you actually want
      to train for and compute loss directly against
    - ``"ipython"`` messages are the return from a tool call

    Chat datasets are multiple rounds of user-assistant messages.
    Instruct datasets are typically a single round with an instruction and model response.
    Tool datasets are a type of chat dataset that includes ipython messages.
    Multimodal datasets are a type of chat dataset that incorporate media into the user messages.

    The :class:`~torchtune.data.Message` forms the core data unit that all tokenizer
    APIs expect. The key component of this class that ensures any dataset is transformed
    into this format is the ``message_transform``. This is a callable class that takes
    in a sample dictionary - typically a single row from the source dataset - that
    processes the sample in any configurable way to output a list of messages::

        [
            Message(
                role=<system|user|assistant|ipython>,
                content=<message>,
            ),
            ...
        ]

    For any custom dataset, use the ``message_transform`` to contain all pre-processing to
    return the list of messages.

    Any model-specific pre-processing that needs to happen can be configured with the ``model_tokenizer``
    parameter. This is another callable class that contains any custom logic tied to the
    model you are fine-tuning and will carry over to inference. For example, text + image
    multimodal datasets requires processing the images in a way specific to the vision
    encoder being used by the model and is agnostic to the specific dataset.

    Tokenization is handled by the ``model_tokenizer``. All :class:`~torchtune.modules.tokenizers.ModelTokenizer`
    can be treated as a ``model_tokenizer`` since it uses the model-specific tokenizer to
    transform the list of messages outputted from the ``message_transform`` into tokens
    used by the model for training. Text-only datasets will simply pass the
    :class:`~torchtune.modules.tokenizers.ModelTokenizer`
    into ``model_tokenizer``. Tokenizers handle prompt templating, if configured.

    Args:
        source: Path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See Hugging Face's ``load_dataset`` for
            more details.
        model_tokenizer: Tokenizer that converts messages to token IDs. Must return
            at minimum ``"tokens"`` and ``"mask"`` keys.
        inference: If True, assistant message content is left empty for generation.
            Default is False.
        deduplicate: Whether to deduplicate consecutive duplicate speech tokens.
        use_modality_tokens: Whether to wrap speech spans with modality boundary tokens.
        filter_fn: Callable used to filter the dataset prior to any pre-processing.
            Default is None.
        train_on_input: Whether to compute loss on user prompt tokens.
        column_map: Mapping from expected column names (``"input"``, ``"output"``) to
            actual column names in the dataset. Default is None.
        new_system_prompt: If specified, prepend a system message. Default is None.
        image_dir: Directory prepended to image paths in the dataset for multimodal
            samples. Default is None.
        additional_keys: Extra dataset columns to pass through to each sample dict.
            Default is ``[]``.
        **load_dataset_kwargs: Additional keyword arguments passed to
            ``datasets.load_dataset``.
    """

    def __init__(
        self,
        *,
        source: str,
        model_tokenizer: Llama3Tokenizer,
        inference: bool = False,
        deduplicate: bool,
        use_modality_tokens: bool,
        filter_fn: Callable | None = None,
        train_on_input: bool,
        column_map: dict[str, str] | None = None,
        new_system_prompt: str | None = None,
        image_dir: Path | None = None,
        additional_keys: list[str] | None = None,
        **load_dataset_kwargs: dict[str, Any],
    ) -> None:
        if additional_keys is None:
            additional_keys = []
        self._message_transform = InputOutputToMessages(
            train_on_input=train_on_input,
            column_map=column_map,
            new_system_prompt=new_system_prompt,
            image_dir=image_dir,
        )
        self._model_tokenizer = model_tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        if not isinstance(self._data, datasets.Dataset):
            raise TypeError(f"Expected a datasets.Dataset object but found {type(self._data)}")
        if any((k in self._data.features) for k in RESERVED_BATCH_KEYS):
            raise ValueError(f"Dataset contains reserved keys: {RESERVED_BATCH_KEYS}")
        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)
        self._inference = inference
        self._deduplicate = deduplicate
        self._use_modality_tokens = use_modality_tokens
        self.additional_keys = additional_keys

    @property
    def inference(self) -> bool:
        return self._inference

    @inference.setter
    def inference(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("inference must be a boolean.")
        self._inference = value

    @property
    def deduplicate(self) -> bool:
        return self._deduplicate

    @deduplicate.setter
    def deduplicate(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("deduplicate must be a boolean.")
        self._deduplicate = value

    @property
    def use_modality_tokens(self) -> bool:
        return self._use_modality_tokens

    @use_modality_tokens.setter
    def use_modality_tokens(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("use_modality_tokens must be a boolean.")
        self._use_modality_tokens = value

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._data[index]
        return self._prepare_sample(sample) | {k: sample[k] for k in self.additional_keys}

    def _prepare_sample(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        transformed_sample = self._message_transform(
            sample,
            deduplicate=self._deduplicate,
            use_modality_tokens=self._use_modality_tokens,
            inference=self._inference,
        )
        if "messages" in transformed_sample:
            validate_messages(transformed_sample["messages"])

        # NOTE Llama3Tokenizer.__call__ actually `pop`s the messages key off so we need to log this before
        LOGGER.debug(f"Messages: \n{transformed_sample['messages']}")

        # NOTE Reminder as of torchtune v0.5.0 tokenizer inference mode is difference between adding or omitting eos_id
        tokenized_dict = self._model_tokenizer(transformed_sample, inference=self._inference)

        LOGGER.debug(f"Tokens: {tokenized_dict['tokens']}")
        LOGGER.debug(f"Mask: {tokenized_dict['mask']}")

        if not ("tokens" in tokenized_dict and "mask" in tokenized_dict):
            keys_str = ", ".join(tokenized_dict.keys())
            error_message = (
                f"model_tokenizer returned the following keys: {keys_str}. Must return 'tokens' and 'mask' as keys."
            )
            raise ValueError(error_message)

        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        tokenized_dict["labels"] = list(
            np.where(
                tokenized_dict["mask"],
                CROSS_ENTROPY_IGNORE_IDX,
                tokenized_dict["tokens"],
            )
        )
        assert len(tokenized_dict["tokens"]) == len(tokenized_dict["labels"])

        return tokenized_dict


class InputOutputToMessages:
    """
    Message transform class that converts a single sample with "input" and "output" fields,
    (or equivalent fields specified in column_map) to user and assistant messages,
    respectively. This is useful for datasets that have two columns, one containing
    the user prompt string and the other containing the model response string::

        |  input          |  output          |
        |-----------------|------------------|
        | "user prompt"   | "model response" |

    Args:
        train_on_input (bool): Whether the model is trained on the user prompt or not.
            Default is False.
        column_map (Optional[dict[str, str]]): a mapping to change the expected "input"
            and "output" column names to the actual column names in the dataset. Keys should
            be "input" and "output" and values should be the actual column names. Default is None,
            keeping the default "input" and "output" column names.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Default is None.
        image_dir (Optional[Path]): path to the directory containing the images that is prepended to all image
            paths in the dataset. For example, if ``image_dir="/home/user/dataset/"` and the sample image path
            was ``"images/1.jpg"``, the final image path that will be loaded is ``"/home/user/dataset/images/1.jpg"``.
            If None, assume images are available in current working directory or are located
            on a remote url. For text-only, leave as None. Default is None.

    Raises:
        ValueError: If ``column_map`` is provided and ``input`` not in ``column_map``, or
            ``output`` not in ``column_map``.
        ValueError: If ``image_dir`` is provided but ``image`` not in ``column_map``.
    """

    def __init__(
        self,
        train_on_input: bool,
        column_map: dict[str, str] | None = None,
        new_system_prompt: str | None = None,
        image_dir: Path | None = None,
    ):
        self.train_on_input = train_on_input
        self.new_system_prompt = new_system_prompt
        if column_map is not None:
            if "input" not in column_map:
                raise ValueError(f"Expected a key of 'input' in column_map but found {column_map.keys()}.")
            if "output" not in column_map:
                raise ValueError(f"Expected a key of 'output' in column_map but found {column_map.keys()}.")
            self.column_map = column_map
        else:
            self.column_map = {"input": "input", "output": "output", "image": "image"}
        # Ensure that if a user seems to want to construct a multimodal transform, they provide a proper column_mapping
        if "image" not in self.column_map and image_dir is not None:
            raise ValueError(
                f"image_dir is specified as {image_dir} but 'image' is not in column_map. "
                "Please specify an 'image' key in column_map."
            )
        self.image_dir = image_dir

    def __call__(
        self,
        sample: Mapping[str, Any],
        *,
        deduplicate: bool,
        use_modality_tokens: bool,
        inference: bool,
    ) -> Mapping[str, Any]:
        is_multimodal = "image" in sample or ("image" in self.column_map and self.column_map["image"] in sample)
        if is_multimodal:
            image_path = sample[self.column_map["image"]]
            if isinstance(image_path, str):
                # Convert image_path to Path obj
                image_path = Path(image_path)
                # If image_dir is not None, prepend image_dir to image_path
                if self.image_dir is not None:
                    image_path = self.image_dir / image_path
                # Load if not loaded
                pil_image = load_image(image_path)
            else:
                pil_image = image_path
            content = [
                {"type": "image", "content": pil_image},
                {"type": "text", "content": sample[self.column_map["input"]]},
            ]
        else:
            sp_tkns = sample[self.column_map["input"]]
            if deduplicate:
                sp_tkns = [k for k, g in groupby(sp_tkns)]
            sp_span = "".join(map(dsu2pua, sp_tkns))
            if use_modality_tokens:
                # NOTE assumes text follows -> reasonable given following tokens are next message header
                sp_span = MODALITY_TOKEN_SPEECH + sp_span + MODALITY_TOKEN_TEXT
            content = [{"type": "text", "content": sp_span}]
        if inference:
            output_content = [{"type": "text", "content": ""}]  # NOTE return empty output for inference i.e. generation
        else:
            output_content = [{"type": "text", "content": sample[self.column_map["output"]]}]
        messages = [
            Message(
                role="user",
                content=content,
                masked=not self.train_on_input,
                eot=True,
            ),
            Message(
                role="assistant",
                content=output_content,
                masked=False,
                eot=True,
            ),
        ]
        if self.new_system_prompt is not None:
            messages = [Message(role="system", content=self.new_system_prompt, masked=True, eot=True), *messages]
        return {"messages": messages}
