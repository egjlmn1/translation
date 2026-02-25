"""Abstract base class for translation models."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Interface that all translation models must implement."""

    @abstractmethod
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full forward pass for training. Returns logits over target vocab."""
        ...

    @abstractmethod
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source sequence. Returns memory tensor."""
        ...

    @abstractmethod
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode one step given memory. Returns logits."""
        ...

    def save_checkpoint(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Save model weights + optional extra state (optimizer, epoch, etc.)."""
        state = {"model_state_dict": self.state_dict()}
        if extra:
            state.update(extra)
        torch.save(state, path)

    def load_checkpoint(self, path: str, device: torch.device = torch.device("cpu")) -> Dict[str, Any]:
        """Load model weights. Returns the full checkpoint dict for restoring optimizer, etc."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
