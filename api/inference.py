"""Translation inference logic."""
import os
from typing import Dict, Optional

import torch

from model.transformer import TransformerModel
from tokenizer.base import BaseTokenizer


class TranslationInference:
    """Handles loading model + tokenizer and running translation."""

    def __init__(
        self,
        model: TransformerModel,
        tokenizer: BaseTokenizer,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def translate(self, text: str, max_len: int = 128) -> Dict[str, str]:
        """
        Translate a single sentence.

        Returns:
            {"input": original, "translation": result}
        """
        # Encode input
        ids = self.tokenizer.encode(text.strip())
        ids = [self.tokenizer.sos_id] + ids[: max_len - 2] + [self.tokenizer.eos_id]
        src = torch.tensor([ids], dtype=torch.long, device=self.device)

        # Greedy decode
        output_ids = self.model.greedy_decode(
            src,
            sos_id=self.tokenizer.sos_id,
            eos_id=self.tokenizer.eos_id,
            max_len=max_len,
        )

        translation = self.tokenizer.decode(output_ids[0].tolist())

        return {
            "input": text,
            "translation": translation,
        }
