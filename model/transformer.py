"""
T5-style Transformer architecture:
- Relative positional encoding (bucketed)
- GeGLU activation in feed-forward
- Pre-Norm with RMSNorm
- Custom head dimensions
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .base import BaseModel


class T5RelativePositionBias(nn.Module):
    def __init__(self, n_heads: int, n_buckets: int = 32, max_distance: int = 128, bidirectional: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.n_buckets = n_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.relative_attention_bias = nn.Embedding(n_buckets, n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """Map relative positions to bucket indices."""
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        # Now n >= 0
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # Safety: avoid log(0) which becomes -inf and crashes the GPU with 'illegal instruction'
        # Safe calculation for the 'large sequence' branch
        # We only apply log to positions where n >= max_exact
        n_float = n.float()
        
        # log_n stays 0 or positive because n_float >= max_exact
        log_n = torch.log(torch.clamp(n_float / max_exact, min=1.0))
        log_max = math.log(max_distance / max_exact)
        
        # Calculate offset within the 'large' buckets
        # offset can range from 0 to (num_buckets - max_exact)
        offset = (log_n / log_max * (num_buckets - max_exact)).to(torch.long)
        
        val_if_large = max_exact + offset
        
        # Force strict upper bound to prevent index overflow
        val_if_large = torch.clamp(val_if_large, max=num_buckets - 1)

        # Merge
        ret += torch.where(is_small, n.to(torch.long), val_if_large)
        return ret

    def forward(self, q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
        """Compute relative position bias matrix (n_heads, q_len, k_len)."""
        # Guard against 0-length sequences which would cause arange to fail
        if q_len == 0 or k_len == 0:
            return torch.zeros((1, self.n_heads, q_len, k_len), device=device)
        
        context_position = torch.arange(q_len, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(k_len, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        
        buckets = self._relative_position_bucket(
            relative_position, self.bidirectional, self.n_buckets, self.max_distance
        )
        
        # (q_len, k_len, n_heads)
        values = self.relative_attention_bias(buckets)

        # REL-BIAS CLAMP: 
        # Prevents extreme values from saturating or breaking the Softmax in attention.
        # This is a key safety net against 'illegal instruction' crashes during long runs.
        values = torch.clamp(values, min=-10.0, max=10.0)
        
        # (q_len, k_len, n_heads) -> (1, n_heads, q_len, k_len)
        values = values.permute([2, 0, 1]).unsqueeze(0) 
        return values


class T5Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner_dim = self.n_heads * self.head_dim
        
        self.q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, d_model, bias=False)
        
        self.dropout_p = dropout

    def forward(
        self, 
        x: torch.Tensor, 
        key_value: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None,
        rel_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: (B, Q, D)
        # key_value: (B, K, D) if cross-attention
        B, Q, _ = x.shape
        kv = key_value if key_value is not None else x
        K = kv.shape[1]

        q = self.q(x).view(B, Q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(kv).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(kv).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)

        # Combine relative bias and mask for scaled_dot_product_attention
        # print(f"DEBUG: q={q.dtype}, mask={mask.dtype if mask is not None else 'None'}, bias={rel_bias.dtype if rel_bias is not None else 'None'}")
        attn_mask = rel_bias
        if mask is not None:
            # Simplify: ensure mask is at least (B, 1, 1, K) or (1, 1, Q, K)
            # Standard broadcasting will handle the rest.
            while mask.dim() < 4:
                mask = mask.unsqueeze(1)
            
            if attn_mask is not None:
                attn_mask = (attn_mask + mask).to(q.dtype)
            else:
                attn_mask = mask.to(q.dtype)
        elif attn_mask is not None:
            attn_mask = attn_mask.to(q.dtype)

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize() 
        #     print("DEBUG: Sync before SDPA passed. Mask shape:", attn_mask.shape if attn_mask is not None else "None")
        
        # if torch.cuda.is_available(): torch.cuda.synchronize()

        # with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            # attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        # if torch.cuda.is_available(): torch.cuda.synchronize()

        out = out.transpose(1, 2).contiguous().view(B, Q, self.inner_dim)
        return self.o(out)


class T5LayerFF(nn.Module):
    """Feed-forward layer with GeGLU (Gated Linear Unit)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # GeGLU uses two weights for the gate
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GeGLU(x) = (wi_0(x) * gelu(wi_1(x)))
        # Using 'tanh' approximation for better BF16 numerical stability.
        gate = F.gelu(self.wi_1(x), approximate='tanh')
        x = self.wi_0(x) * gate
        x = self.dropout(x)
        return self.wo(x)


class TransformerModel(BaseModel):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 6,
        n_encoder_layers: int = 8,
        n_decoder_layers: int = 4,
        d_ff: int = 1024,
        head_dim: int = 64,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.debug_sync = False  # Set to True to insert CUDA syncs between every op
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # Weight Tying: Share weights between src_embedding, tgt_embedding and output_projection
        if src_vocab_size == tgt_vocab_size:
            print("[TransformerModel] Weight tying enabled")
            self.tgt_embedding.weight = self.src_embedding.weight
            self.output_projection.weight = self.src_embedding.weight
        
        # Shared relative biases
        self.enc_rel_bias = T5RelativePositionBias(n_heads, bidirectional=True)
        self.dec_rel_bias = T5RelativePositionBias(n_heads, bidirectional=False) # Causal
        
        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                "norm_attn": nn.RMSNorm(d_model),
                "attn": T5Attention(d_model, n_heads, head_dim=head_dim, dropout=dropout),
                "norm_ff": nn.RMSNorm(d_model),
                "ff": T5LayerFF(d_model, d_ff, dropout)
            }) for _ in range(n_encoder_layers)
        ])
        self.enc_final_norm = nn.RMSNorm(d_model)
        
        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            nn.ModuleDict({
                "norm_self_attn": nn.RMSNorm(d_model),
                "self_attn": T5Attention(d_model, n_heads, head_dim=head_dim, dropout=dropout),
                "norm_cross_attn": nn.RMSNorm(d_model),
                "cross_attn": T5Attention(d_model, n_heads, head_dim=head_dim, dropout=dropout),
                "norm_ff": nn.RMSNorm(d_model),
                "ff": T5LayerFF(d_model, d_ff, dropout)
            }) for _ in range(n_decoder_layers)
        ])
        self.dec_final_norm = nn.RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _dsync(self, label: str) -> None:
        """Debug sync: if debug_sync is enabled, synchronize CUDA and print the label.
        If the crash happens right AFTER a label, the PREVIOUS label's op is the culprit."""
        if self.debug_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
            print(f"  [DSYNC OK] {label}")

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=self.d_model ** -0.5)

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones((sz, sz), device=device)).transpose(0, 1)
        # Use a large finite negative number instead of -inf for BF16 stability
        mask = mask.float().masked_fill(mask == 0, -30000.0).masked_fill(mask == 1, 0.0)
        return mask.unsqueeze(0).unsqueeze(0)

    def _make_padding_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        pad_mask = (tokens == self.pad_id)
        # Standard broadcasting shape: (B, 1, 1, S)
        mask = pad_mask.float().masked_fill(pad_mask, -30000.0).masked_fill(~pad_mask, 0.0).unsqueeze(1).unsqueeze(1)
        
        # SAFETY: Ensure no row is 100% masked for any sequence.
        # We force the first token (usually SOS or first word) to always be visible.
        # This prevents Softmax(all -inf) = NaN, which causes GPU 'illegal instruction' crashes.
        mask[..., 0] = 0.0 
        
        return mask

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # src: (B, S)
        # T5-style: embeddings are scaled by sqrt(d_model)
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        self._dsync("enc_embedding")
        x = self.dropout(x)
        
        # Relative bias: (1, heads, S, S)
        rel_bias = self.enc_rel_bias(src.size(1), src.size(1), src.device)
        self._dsync("enc_rel_bias")

        for i, layer in enumerate(self.encoder_layers):
            normed = layer["norm_attn"](x)
            self._dsync(f"enc_layer{i}_norm_attn")
            x = x + layer["attn"](normed, mask=src_mask, rel_bias=rel_bias)
            self._dsync(f"enc_layer{i}_self_attn")
            
            normed = layer["norm_ff"](x)
            self._dsync(f"enc_layer{i}_norm_ff")
            x = x + layer["ff"](normed)
            self._dsync(f"enc_layer{i}_ff")
            
        return self.enc_final_norm(x)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # tgt: (B, T), memory: (B, S, D)
        # Embeddings are scaled by sqrt(d_model)
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        self._dsync("dec_embedding")
        x = self.dropout(x)
        
        self_rel_bias = self.dec_rel_bias(tgt.size(1), tgt.size(1), tgt.device)
        self._dsync("dec_rel_bias")
        
        for i, layer in enumerate(self.decoder_layers):
            normed = layer["norm_self_attn"](x)
            self._dsync(f"dec_layer{i}_norm_self_attn")
            x = x + layer["self_attn"](normed, mask=tgt_mask, rel_bias=self_rel_bias)
            self._dsync(f"dec_layer{i}_self_attn")
            
            normed = layer["norm_cross_attn"](x)
            self._dsync(f"dec_layer{i}_norm_cross_attn")
            x = x + layer["cross_attn"](normed, key_value=memory, mask=memory_mask)
            self._dsync(f"dec_layer{i}_cross_attn")
            
            normed = layer["norm_ff"](x)
            self._dsync(f"dec_layer{i}_norm_ff")
            x = x + layer["ff"](normed)
            self._dsync(f"dec_layer{i}_ff")
            
        return self.dec_final_norm(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. SEQUENCE LENGTH GUARDS
        # Prevention: Zero-length sequences cause Softmax NaNs and kernel crashes
        if src.size(1) == 0 or tgt.size(1) == 0:
            raise ValueError(f"[Model] Received empty sequence: src_len={src.size(1)}, tgt_len={tgt.size(1)}")

        # Resource Safety: Prevents OOM and ensures consistency with T5 relative buckets
        if src.size(1) > self.max_seq_len or tgt.size(1) > self.max_seq_len:
            raise ValueError(
                f"[Model] Sequence length {max(src.size(1), tgt.size(1))} exceeds max_seq_len {self.max_seq_len}"
            )

        if src_padding_mask is None:
            src_padding_mask = self._make_padding_mask(src) # (B, 1, 1, S)
        if tgt_padding_mask is None:
            tgt_padding_mask = self._make_padding_mask(tgt) # (B, 1, 1, T)
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device) # (1, 1, T, T)
        
        # Combined mask for decoder: (B, 1, 1, T) + (1, 1, T, T) -> (B, 1, T, T)
        # Using addition ensures that if either mask is -inf, the result is -inf.
        full_tgt_mask = tgt_padding_mask + tgt_mask
        self._dsync("forward_masks_built")
        
        memory = self.encode(src, src_padding_mask)
        self._dsync("forward_encode_done")
        output = self.decode(tgt, memory, tgt_mask=full_tgt_mask, memory_mask=src_padding_mask)
        self._dsync("forward_decode_done")
        
        return self.output_projection(output)

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, sos_id: int, eos_id: int, max_len: int = 128) -> torch.Tensor:
        self.eval()
        batch_size = src.size(0)
        src_mask = self._make_padding_mask(src)
        memory = self.encode(src, src_mask)
        
        # Initialize with batch_size
        ys = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=src.device)
        
        for i in range(max_len):
            # Target mask for current sequence
            tm = self.generate_square_subsequent_mask(ys.size(1), ys.device)
            out = self.decode(ys, memory, tgt_mask=tm, memory_mask=src_mask)
            logits = self.output_projection(out)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            
            # Simple early exit for single batch or if all batches would hit EOS
            if batch_size == 1 and next_token.item() == eos_id: 
                break
        return ys
