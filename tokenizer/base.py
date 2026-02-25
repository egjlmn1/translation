"""Abstract base class for tokenizers."""
from abc import ABC, abstractmethod
from typing import List


class BaseTokenizer(ABC):
    """Interface that all tokenizers must implement."""

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}

    # Special token strings
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    @abstractmethod
    def train(self, corpus: List[str], vocab_size: int) -> None:
        """Train the tokenizer on a corpus of text."""
        ...

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to a list of token IDs."""
        ...

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text to a list of token strings."""
        ...

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts. Default implementation loops; override for speed."""
        return [self.encode(t) for t in texts]

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back to text."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Save tokenizer state to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load tokenizer state from disk."""
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        ...

    @property
    def pad_id(self) -> int:
        return self.token_to_id.get(self.PAD_TOKEN, 0)

    @property
    def sos_id(self) -> int:
        return self.token_to_id.get(self.SOS_TOKEN, 1)

    @property
    def eos_id(self) -> int:
        return self.token_to_id.get(self.EOS_TOKEN, 2)

    @property
    def unk_id(self) -> int:
        return self.token_to_id.get(self.UNK_TOKEN, 3)
