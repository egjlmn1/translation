"""Replay a saved crash batch with debug_sync enabled to pinpoint the exact CUDA op."""
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import TransformerModel
from tokenizer.bpe import BPETokenizer


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load config
    with open(os.path.join(project_root, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(os.path.join(project_root, config["tokenizer"]["save_dir"]))

    # Create model (on CPU first)
    mc = config["model"]
    model = TransformerModel(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=mc["d_model"],
        n_heads=mc["n_heads"],
        n_encoder_layers=mc["n_encoder_layers"],
        n_decoder_layers=mc["n_decoder_layers"],
        d_ff=mc["d_ff"],
        head_dim=mc.get("head_dim", 64),
        dropout=mc["dropout"],
        max_seq_len=mc["max_seq_len"],
        pad_id=tokenizer.pad_id,
    )

    # Load checkpoint
    ckpt_path = os.path.join(project_root, "checkpoints", "checkpoint_latest.pt")
    if os.path.exists(ckpt_path):
        model.load_checkpoint(ckpt_path, device=torch.device("cpu"))
        print(f"[Replay] Loaded checkpoint from {ckpt_path}")

    # Load crash batch
    crash_path = os.path.join(project_root, "checkpoints", "debug_crash_batch.pt")
    if not os.path.exists(crash_path):
        print(f"[Replay] No crash batch found at {crash_path}")
        print(f"         Run training until the crash happens first.")
        return

    crash = torch.load(crash_path, weights_only=False)
    src = crash["src"]
    tgt = crash["tgt"]
    print(f"\n[Replay] Crash batch from step {crash['global_step']}, epoch {crash['epoch']}")
    print(f"  src shape: {src.shape}, range: [{src.min().item()}, {src.max().item()}]")
    print(f"  tgt shape: {tgt.shape}, range: [{tgt.min().item()}, {tgt.max().item()}]")
    print(f"  vocab_size: {tokenizer.vocab_size}")

    # Check for OOB tokens
    oob_src = (src >= tokenizer.vocab_size) | (src < 0)
    oob_tgt = (tgt >= tokenizer.vocab_size) | (tgt < 0)
    if oob_src.any():
        print(f"  ⚠ OOB tokens in src: {src[oob_src].tolist()}")
    if oob_tgt.any():
        print(f"  ⚠ OOB tokens in tgt: {tgt[oob_tgt].tolist()}")

    # === STEP 1: Try on CPU (gives clean Python errors) ===
    print(f"\n{'='*60}")
    print("[Replay] STEP 1: Running on CPU (clean error messages)...")
    print(f"{'='*60}")
    model.eval()
    tgt_input = tgt[:, :-1]
    try:
        with torch.no_grad():
            logits = model(src, tgt_input)
        print("  ✓ CPU forward pass succeeded!")
    except Exception as e:
        print(f"  ✗ CPU error: {e}")
        return

    # === STEP 2: Try on GPU with debug_sync ===
    if torch.cuda.is_available():
        print(f"\n{'='*60}")
        print("[Replay] STEP 2: Running on GPU with debug_sync=True...")
        print("  (Last [DSYNC OK] line before crash = the guilty operation)")
        print(f"{'='*60}")
        device = torch.device("cuda")
        model = model.to(device)
        model.debug_sync = True
        src_gpu = src.to(device)
        tgt_input_gpu = tgt_input.to(device)
        try:
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                logits = model(src_gpu, tgt_input_gpu)
            torch.cuda.synchronize()
            print("  ✓ GPU forward pass succeeded!")
            print("  (The crash may be non-deterministic — try running multiple times)")
        except Exception as e:
            print(f"\n  ✗ GPU crash after the last [DSYNC OK] line above!")
            print(f"    Error: {e}")
    else:
        print("[Replay] No GPU available, skipping GPU replay.")

    # === STEP 3: Check model weights ===
    print(f"\n{'='*60}")
    print("[Replay] STEP 3: Model weight health check")
    print(f"{'='*60}")
    for name, p in model.named_parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            nan_c = torch.isnan(p).sum().item()
            inf_c = torch.isinf(p).sum().item()
            print(f"  ⚠ {name}: nan={nan_c}, inf={inf_c}")
    print("  Done.")


if __name__ == "__main__":
    main()
