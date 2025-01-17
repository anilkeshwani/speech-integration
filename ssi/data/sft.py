from typing import Any, Callable, Mapping, Optional

import numpy as np
from datasets import load_dataset
from sardalign.utils import dsu2pua
from torch.utils.data import Dataset
from torchtune.data import Message, PromptTemplate, Role
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.data._messages import validate_messages
from torchtune.models.llama3 import Llama3Tokenizer
from torchtune.modules.transforms import Transform


class SFTDataset(Dataset):
    """
    Creating SFT dataset from Hugging Face Hub, local files, or remote files.
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
    used by the model for training. Text-only datasets will simply pass the :class:`~torchtune.modules.tokenizers.ModelTokenizer`
    into ``model_tokenizer``. Tokenizers handle prompt templating, if configured.

    Args:
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details.
        message_transform (Transform): callable that keys into the desired fields in the sample
            and converts text content to a list of :class:`~torchtune.data.Message`. It is expected that the final list
            of messages are stored in the ``"messages"`` key.
        model_tokenizer (Transform): callable that applies model-specific pre-processing to the sample after the list of
            messages is created from ``message_transform``. This includes tokenization and any modality-specific
            transforms. It is expected to return at minimum ``"tokens"`` and ``"mask"`` keys.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        **load_dataset_kwargs (dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details.
    """

    def __init__(
        self,
        *,
        source: str,
        message_transform: Transform,
        model_tokenizer: Llama3Tokenizer,
        inference: bool = False,
        filter_fn: Optional[Callable] = None,
        **load_dataset_kwargs: dict[str, Any],
    ) -> None:
        self._message_transform = message_transform
        self._model_tokenizer = model_tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)
        self._inference = inference

    @property
    def inference(self) -> bool:
        return self._inference

    @inference.setter
    def inference(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("inference must be a boolean.")
        self._inference = value

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], Any]:
        sample = self._data[index]
        return self._prepare_sample(sample), sample

    def _prepare_sample(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        transformed_sample = self._message_transform(sample)
        if "messages" in transformed_sample:
            validate_messages(transformed_sample["messages"])

        tokenized_dict = self._model_tokenizer(transformed_sample)

        if not ("tokens" in tokenized_dict and "mask" in tokenized_dict):
            keys_str = ", ".join(tokenized_dict.keys())
            error_message = (
                "model_tokenizer returned the following keys: " f"{keys_str}. Must return 'tokens' and 'mask' as keys."
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


class ASRInputOutputToMessages(Transform):
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

    Raises:
        ValueError: If ``column_map`` is provided and ``input`` not in ``column_map``, or
            ``output`` not in ``column_map``.
    """

    def __init__(
        self,
        train_on_input: bool,
        column_map: Optional[dict[str, str]] = None,
        new_system_prompt: Optional[str] = None,
    ):
        self.train_on_input = train_on_input
        self.new_system_prompt = new_system_prompt
        if column_map:
            if "input" not in column_map:
                raise ValueError(f"Expected a key of 'input' in column_map but found {column_map.keys()}.")
            if "output" not in column_map:
                raise ValueError(f"Expected a key of 'output' in column_map but found {column_map.keys()}.")
            self._column_map = column_map
        else:
            self._column_map = {"input": "input", "output": "output"}

    def __call__(self, sample: Mapping[str, Any], inference: bool) -> Mapping[str, Any]:
        messages = [
            Message(
                role="user",
                content="".join(map(dsu2pua, sample[self._column_map["input"]])),
                masked=not self.train_on_input,
                eot=True,
            ),
            Message(
                role="assistant",
                # NOTE at inference, no output return empty string for output
                content=sample[self._column_map["output"]] if not inference else "",
                masked=False,
                eot=True,
            ),
        ]
        if self.new_system_prompt is not None:
            messages = [Message(role="system", content=self.new_system_prompt, masked=True, eot=True)] + messages
        return {"messages": messages}
