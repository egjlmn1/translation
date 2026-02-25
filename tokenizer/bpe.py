"""BPE tokenizer using HuggingFace's `tokenizers` library (Rust-based, very fast)."""
import json
import os
from typing import Dict, List, Optional

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from .base import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """
    BPE tokenizer backed by HuggingFace's `tokenizers` library.
    Trains from scratch on your corpus — no pretrained anything.
    Uses Rust under the hood so training is ~100x faster than pure Python.
    """

    def __init__(self) -> None:
        self._tokenizer: Optional[Tokenizer] = None
        self.token_to_id: Dict[str, int] = {}
        self._vocab_size: int = 0

    def train(self, corpus: List[str], vocab_size: int, min_frequency: int = 2) -> None:
        """
        Train BPE on *corpus* (list of raw strings).
        """
        # Build a BPE tokenizer
        tokenizer = Tokenizer(models.BPE())

        # Pre-tokenize on whitespace + punctuation (similar to GPT-2)
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Decoder to reverse byte-level encoding
        tokenizer.decoder = decoders.ByteLevel()

        # Special tokens
        special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]

        # Trainer
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True,
        )

        # Train from iterator (no need to write temp files)
        print(f"[BPE] Training on {len(corpus)} texts, target vocab_size={vocab_size}...")
        tokenizer.train_from_iterator(corpus, trainer=trainer)

        # Post-processor: add SOS/EOS automatically
        sos_id = tokenizer.token_to_id(self.SOS_TOKEN)
        eos_id = tokenizer.token_to_id(self.EOS_TOKEN)
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.SOS_TOKEN} $A {self.EOS_TOKEN}",
            special_tokens=[
                (self.SOS_TOKEN, sos_id),
                (self.EOS_TOKEN, eos_id),
            ],
        )

        # NOTE: We do NOT call tokenizer.enable_padding() here.
        # We handle padding inside the Data Loader's collate_fn or sorting logic,
        # which allows for more efficient bucketing.

        self._tokenizer = tokenizer
        self.token_to_id = tokenizer.get_vocab()
        self._vocab_size = tokenizer.get_vocab_size()
        print(f"[BPE] Training complete. Vocab size: {self._vocab_size}")

    def encode(self, text: str) -> List[int]:
        """Encode text to a list of token IDs (includes SOS/EOS)."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not trained or loaded.")
        encoding = self._tokenizer.encode(text)
        return encoding.ids

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text to a list of token strings (includes SOS/EOS)."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not trained or loaded.")
        encoding = self._tokenizer.encode(text)
        return encoding.tokens

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts in parallel using Rust backend."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not trained or loaded.")
        encodings = self._tokenizer.encode_batch(texts)
        return [e.ids for e in encodings]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text (strips special tokens)."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not trained or loaded.")
        # Use the built-in decode with special token skipping
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def save(self, path: str) -> None:
        """Save tokenizer to a JSON file."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not trained.")
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, "bpe_tokenizer.json")
        self._tokenizer.save(filepath)
        print(f"[BPE] Saved tokenizer to {filepath}")

    def load(self, path: str) -> None:
        """Load tokenizer from a JSON file."""
        filepath = os.path.join(path, "bpe_tokenizer.json")
        self._tokenizer = Tokenizer.from_file(filepath)
        # Ensure padding is disabled so encode_batch doesn't pad automatically
        self._tokenizer.no_padding()
        
        self.token_to_id = self._tokenizer.get_vocab()
        self._vocab_size = self._tokenizer.get_vocab_size()
        print(f"[BPE] Loaded tokenizer from {filepath} (vocab={self._vocab_size})")

    @property
    def vocab_size(self) -> int:
        return self._vocab_size
