from torchtune.data import PromptTemplate
from torchtune.models.llama3 import Llama3Tokenizer
from torchtune.models.llama3._tokenizer import CL100K_PATTERN
from torchtune.modules.tokenizers import TikTokenBaseTokenizer


CL100K_PATTERN_PUA = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}\p{Co}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}\p{Co}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+|\p{Co}"""  # noqa

assert CL100K_PATTERN_PUA != CL100K_PATTERN


class Llama3TokenizerPUA(Llama3Tokenizer):
    def __init__(
        self,
        path: str,
        special_tokens: dict[str, int] | None = None,
        max_seq_len: int | None = None,
        prompt_template: PromptTemplate | None = None,
    ):
        super().__init__(path, special_tokens, max_seq_len, prompt_template)
        # possible to monkey patch as self.tt_model does not affect subsequent initialization code in Llama3Tokenizer
        self.tt_model = TikTokenBaseTokenizer(
            path=path,
            name="llama3_tiktoken",
            pattern=CL100K_PATTERN_PUA,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            special_tokens=self.special_tokens,
        )
