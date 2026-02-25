import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import TransformerModel

def test_t5_architecture():
    print("Testing T5-small style architecture...")
    
    # Matching your config.yaml
    d_model = 512
    n_heads = 6
    n_enc = 8
    n_dec = 4
    d_ff = 1024
    vocab_size = 32000
    
    print(f"Params: d_model={d_model}, n_heads={n_heads}, En={n_enc}, De={n_dec}, d_ff={d_ff}")
    
    model = TransformerModel(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_enc,
        n_decoder_layers=n_dec,
        d_ff=d_ff
    )
    
    # Dummy data
    batch_size = 2
    src_len = 10
    tgt_len = 12
    
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))
    
    try:
        # Forward pass
        logits = model(src, tgt)
        print(f"Forward success! Logits shape: {logits.shape}")
        
        # Test greedy decode
        decoded = model.greedy_decode(src, sos_id=1, eos_id=2, max_len=5)
        print(f"Decode success! Decoded shape: {decoded.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_t5_architecture()
