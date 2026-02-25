import pandas as pd
import argparse
import time
import sys
import os
from tqdm import tqdm

# Add project root to path so we can import from training, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_translator import GoogleTranslator
from training.metrics import compute_bleu
import yaml
from concurrent.futures import ThreadPoolExecutor

def validate_dataset(start_row, end_row, threshold, config_path="config.yaml", override_csv=None, output_path=None, num_threads=5):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    csv_path = override_csv if override_csv else config["data"]["csv_path"]
    src_col = config["data"]["src_col"]
    tgt_col = config["data"]["tgt_col"]

    if not os.path.exists(csv_path):
        print(f"Error: Dataset file not found at {csv_path}")
        return

    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path, skiprows=range(1, start_row) if start_row > 0 else None, nrows=end_row - start_row + 1)
    
    en_to_fr = GoogleTranslator(source='en', target='fr')
    fr_to_en = GoogleTranslator(source='fr', target='en')
    
    good_rows = []
    bad_count = 0

    print(f"Validating rows {start_row} to {end_row} with threshold {threshold} using {num_threads} threads...")
    if output_path:
        print(f"Good sentences will be saved to: {output_path}")
    
    def validate_row(args):
        idx, row = args
        en_text = str(row[src_col]).strip()
        fr_text = str(row[tgt_col]).strip()
        
        if not en_text or not fr_text:
            return None

        try:
            # EN -> FR and FR -> EN
            google_fr = en_to_fr.translate(en_text)
            google_en = fr_to_en.translate(fr_text)

            bleu_fr = compute_bleu(google_fr, fr_text) * 100
            bleu_en = compute_bleu(google_en, en_text) * 100
            avg_bleu = (bleu_fr + bleu_en) / 2
            
            if avg_bleu >= threshold:
                return (True, row, avg_bleu)
            else:
                print(f"  EN: {en_text}")
                print(f"  FR: {fr_text}")
                print(f"  Google EN: {google_en}")
                print(f"  Google FR: {google_fr}")
                return (False, row, avg_bleu, google_fr, google_en)
        except Exception as e:
            return (None, row, str(e))

    # Using ThreadPoolExecutor for concurrent requests
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Prepare arguments for each row
        row_args = [(i, row) for i, row in df.iterrows()]
        
        # Wrap with tqdm for progress bar
        for res in tqdm(executor.map(validate_row, row_args), total=len(row_args)):
            if res is None:
                continue
            
            success = res[0]
            if success is True:
                good_rows.append(res[1])
            elif success is False:
                bad_count += 1
                # Optional: print failure details if needed
            else: # Error case
                pass

    print("\n" + "="*50)
    print(f"Validation complete.")
    print(f"Total processed: {len(df)}")
    print(f"Passed: {len(good_rows)}")
    print(f"Failed: {bad_count}")
    print("="*50)

    if output_path and good_rows:
        good_df = pd.DataFrame(good_rows)
        good_df.to_csv(output_path, index=False)
        print(f"Saved {len(good_rows)} clean sentences to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate dataset rows and filter good ones")
    parser.add_argument("--start", type=int, default=0, help="Start row index")
    parser.add_argument("--end", type=int, default=100, help="End row index")
    parser.add_argument("--threshold", type=float, default=20.0, help="BLEU score threshold (0-100)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file (overrides config)")
    parser.add_argument("--output", type=str, default="data/validated_clean.csv", help="Path to save good sentences")
    parser.add_argument("--threads", type=int, default=5, help="Number of concurrent translation threads")

    args = parser.parse_args()
    validate_dataset(args.start, args.end, args.threshold, args.config, args.csv, args.output, args.threads)
