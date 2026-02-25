"""Pretrained tokenizer wrapper for Hugging Face models."""
import os
from typing import List, Optional

from tokenizers import Tokenizer
from .base import BaseTokenizer

class PretrainedTokenizer(BaseTokenizer):
    """
    Wraps a pretrained Hugging Face tokenizer.
    Loads from a model ID (e.g. 'xlm-roberta-base') or a local directory.
    This allows swapping your custom BPE for a state-of-the-art one.
    """

    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-fr"):
        super().__init__()
        self.model_name = model_name
        self._tokenizer = None
        self._vocab_size = 0
        
        try:
            from transformers import AutoTokenizer
            # use_fast=False is often more stable for MarianMT/SentencePiece models
            self._hf_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            self._setup_metadata()
        except Exception as e:
            print(f"[Pretrained] Warning: Could not initialize {model_name}: {e}")

    def _setup_metadata(self):
        """Map Hugging Face special tokens to our internal names."""
        if self._hf_tokenizer:
            self.token_to_id = self._hf_tokenizer.get_vocab()
            self._vocab_size = len(self.token_to_id)
            
            # Common Hugging Face special token attributes
            mapping = {
                self._hf_tokenizer.bos_token: self.SOS_TOKEN,
                self._hf_tokenizer.eos_token: self.EOS_TOKEN,
                self._hf_tokenizer.unk_token: self.UNK_TOKEN,
                self._hf_tokenizer.pad_token: self.PAD_TOKEN,
            }
            
            for hf_token, our_token in mapping.items():
                if hf_token:
                    self.token_to_id[our_token] = self.token_to_id.get(hf_token)

    def train(self, corpus: List[str], vocab_size: int) -> None:
        """Pretrained tokenizers are already trained."""
        print(f"[Pretrained] No training needed for {self.model_name}.")

    def encode(self, text: str) -> List[int]:
        if self._hf_tokenizer is None:
            raise RuntimeError(f"Tokenizer {self.model_name} not loaded.")
        return self._hf_tokenizer.encode(text, add_special_tokens=True)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        if self._hf_tokenizer is None:
            raise RuntimeError(f"Tokenizer {self.model_name} not loaded.")
        # transformers handle batching well
        return self._hf_tokenizer(texts, add_special_tokens=True)["input_ids"]

    def decode(self, ids: List[int]) -> str:
        if self._hf_tokenizer is None:
            raise RuntimeError(f"Tokenizer {self.model_name} not loaded.")
        return self._hf_tokenizer.decode(ids, skip_special_tokens=True)

    def save(self, path: str) -> None:
        if self._hf_tokenizer:
            self._hf_tokenizer.save_pretrained(path)

    def load(self, path: str) -> None:
        from transformers import AutoTokenizer
        self._hf_tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        self._setup_metadata()

    @property
    def vocab_size(self) -> int:
        return self._vocab_size
