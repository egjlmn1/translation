"""Script to test the tokenizer: reconstruction, fertility, and length distribution."""
import argparse
import os
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.bpe import BPETokenizer
from tokenizer.pretrained import PretrainedTokenizer

def test_tokenizer(config_path, tok_type="bpe", max_test_rows=None):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load tokenizer
    if tok_type == "pretrained":
        model_name = config["tokenizer"].get("model_name", "xlm-roberta-base")
        print(f"[Tester] Using Pretrained Tokenizer ({model_name})...")
        tokenizer = PretrainedTokenizer(model_name)
    else:
        print(f"[Tester] Using Custom BPE Tokenizer...")
        tokenizer = BPETokenizer()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tok_dir = os.path.join(project_root, config["tokenizer"]["save_dir"])
    
    # Custom BPE must be loaded from local files if they exist
    if tok_type == "bpe":
        if not os.path.exists(os.path.join(tok_dir, "bpe_tokenizer.json")):
            print(f"[Tester] ERROR: BPE Tokenizer not found in {tok_dir}. Train it first.")
            return
        tokenizer.load(tok_dir)
    elif os.path.exists(os.path.join(tok_dir, "tokenizer.json")):
        # Pretrained can also be loaded from local cache if we saved it
        tokenizer.load(tok_dir)
    
    # Load CSV in chunks
    csv_path = os.path.join(project_root, config["data"]["csv_path"])
    src_col = config["data"]["src_col"]
    tgt_col = config["data"]["tgt_col"]
    chunk_size = 20000
    
    # Statistics
    total_samples = 0
    mismatches = 0
    total_tokens = 0
    total_words = 0
    lengths = []
    
    print(f"[Tester] Loading data from {csv_path}...")
    if max_test_rows:
        print(f"[Tester] Limit set to {max_test_rows} rows.")
    else:
        print(f"[Tester] Processing ENTIRE file (this may take a while as the file is ~8GB)...")
    
    start_time = time.time()
    
    try:
        # We use chunksize because the file is massive (8GB)
        it = pd.read_csv(csv_path, chunksize=chunk_size, on_bad_lines='skip')
        
        for chunk_idx, df in enumerate(it):
            for _, row in df.iterrows():
                total_samples += 1
                
                for col in [src_col, tgt_col]:
                    text = str(row[col])
                    if not text or text == "nan":
                        continue
                    
                    # 1. Fertility (Words)
                    words = text.split()
                    total_words += len(words)
                    
                    # 2. Encode
                    ids = tokenizer.encode(text)
                    # Note: ids includes <sos> and <eos> (2 extra tokens)
                    token_count = len(ids)
                    total_tokens += (token_count - 2)
                    lengths.append(token_count)
                    
                    # 3. Decode
                    decoded = tokenizer.decode(ids)
                    
                    # 4. Reconstruction Check
                    # We normalize whitespace for the comparison as BPE pre-tokenizers often consolidate spaces
                    orig_norm = " ".join(text.split())
                    deco_norm = " ".join(decoded.split())
                    
                    if orig_norm != deco_norm:
                        # Character-level check (ignoring case/accents? No, should be exact)
                        mismatches += 1
                        if mismatches <= 3:
                            print(f"\n[Tester] Reconstruction Mismatch at row {total_samples}, column '{col}':")
                            print(f"  Original:  {text[:100]}{'...' if len(text)>100 else ''}")
                            print(f"  Decoded:   {decoded[:100]}{'...' if len(decoded)>100 else ''}")
                            print(f"  Difference found in normalized form.")

                if total_samples % 10000 == 0:
                    elapsed = time.time() - start_time
                    speed = total_samples / elapsed
                    print(f"[Tester] Processed {total_samples} samples... ({speed:.1f} rows/s)")

                if max_test_rows and total_samples >= max_test_rows:
                    break
            
            if max_test_rows and total_samples >= max_test_rows:
                break
                
    except KeyboardInterrupt:
        print("\n[Tester] Interrupted by user. Processing collected stats...")
    except Exception as e:
        print(f"\n[Tester] Error during processing: {e}")

    if total_samples == 0:
        print("[Tester] No data was processed.")
        return

    if not lengths:
        print("[Tester] No tokens were successfully extracted. Check for error messages above.")
        return

    # Calculate results
    total_tests = total_samples * 2
    reconstruction_accuracy = 100 * (1 - mismatches / total_tests)
    fertility = total_tokens / total_words if total_words > 0 else 0
    
    print("\n" + "="*50)
    print(" TOKENIZER TEST REPORT")
    print("="*50)
    print(f"Rows Processed:        {total_samples:,}")
    print(f"Total Tests:           {total_tests:,}")
    print(f"Mismatches:            {mismatches:,}")
    print(f"Reconstruction Acc:    {reconstruction_accuracy:.4f}%")
    print(f"Total Words:           {total_words:,}")
    print(f"Total Tokens:          {total_tokens:,} (excluding special tokens)")
    print(f"Fertility Rate:        {fertility:.4f} (tokens/word)")
    print("-" * 50)
    
    # Sequence Length Stats (with SOS/EOS)
    lens = np.array(lengths)
    p50 = np.percentile(lens, 50)
    p95 = np.percentile(lens, 95)
    p99 = np.percentile(lens, 99)
    max_l = np.max(lens)
    
    print(f"Sequence Lengths (including <sos>/<eos>):")
    print(f"  Median:              {int(p50)}")
    print(f"  95th Percentile:     {int(p95)}")
    print(f"  99th Percentile:     {int(p99)}")
    print(f"  Maximum:             {max_l}")
    
    model_limit = config["model"]["max_seq_len"]
    over_limit = np.sum(lens > model_limit)
    if over_limit > 0:
        print(f"  !!! WARNING: {over_limit:,} sequences ({100*over_limit/len(lens):.2f}%) exceed model limit of {model_limit}")
    else:
        print(f"  All sequences within model limit of {model_limit}.")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    # Filter for reasonable view
    plot_lens = lens[lens < 300]
    plt.hist(plot_lens, bins=range(0, 301, 2), color='#6c5ce7', edgecolor='white', alpha=0.8)
    
    plt.axvline(p99, color='#ff7675', linestyle='dashed', linewidth=2, label=f'99th percentile ({int(p99)})')
    plt.axvline(model_limit, color='#fdcb6e', linestyle='--', linewidth=2, label=f'Model limit ({model_limit})')
    
    plt.title("Distribution of Token Sequence Lengths", fontsize=14, fontweight='bold')
    plt.xlabel("Sequence Length (Number of Tokens)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.legend()
    
    # Extra aesthetics
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    output_img = "tokenizer_test_results.png"
    plt.savefig(output_img, dpi=150, bbox_inches='tight')
    print(f"\n[Tester] Histogram plot saved to: {output_img}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test tokenizer reconstruction and fertility.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--type", default="bpe", choices=["bpe", "pretrained"], help="Tokenizer type to test")
    parser.add_argument("--max-rows", type=int, default=100000, help="Max rows to test (default 100k, set 0 for all)")
    args = parser.parse_args()
    
    rows = args.max_rows if args.max_rows > 0 else None
    test_tokenizer(args.config, tok_type=args.type, max_test_rows=rows)
