"""Train the BPE tokenizer on the translation corpus."""
import os
import sys

import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.bpe import BPETokenizer


def main():
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config["data"]["csv_path"])
    tok_config = config["tokenizer"]
    sample_size = tok_config["sample_size"]
    vocab_size = tok_config["vocab_size"]
    min_freq = tok_config.get("min_frequency", 2)
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), tok_config["save_dir"])

    print(f"[train_tokenizer] Reading up to {sample_size} rows from {csv_path}...")

    # Stream sample_size rows
    corpus = []
    rows_read = 0
    src_col = config["data"]["src_col"]
    tgt_col = config["data"]["tgt_col"]

    reader = pd.read_csv(
        csv_path,
        chunksize=50_000,
        usecols=[src_col, tgt_col],
        on_bad_lines="skip",
        encoding="utf-8",
    )

    for chunk in reader:
        chunk = chunk.dropna()
        for _, row in chunk.iterrows():
            src = str(row[src_col]).strip()
            tgt = str(row[tgt_col]).strip()
            if src:
                corpus.append(src)
            if tgt:
                corpus.append(tgt)
            rows_read += 1
            if rows_read >= sample_size:
                break
        if rows_read >= sample_size:
            break

    print(f"[train_tokenizer] Collected {len(corpus)} texts from {rows_read} rows.")

    # Train
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=vocab_size, min_frequency=min_freq)

    # Save
    tokenizer.save(save_dir)
    print(f"[train_tokenizer] Done! Tokenizer saved to {save_dir}/")

    # Quick test
    test_en = "Hello, how are you?"
    test_fr = "Bonjour, comment allez-vous?"
    enc_en = tokenizer.encode(test_en)
    enc_fr = tokenizer.encode(test_fr)
    print(f"\n  Test EN: '{test_en}' -> {enc_en} -> '{tokenizer.decode(enc_en)}'")
    print(f"  Test FR: '{test_fr}' -> {enc_fr} -> '{tokenizer.decode(enc_fr)}'")


if __name__ == "__main__":
    main()
