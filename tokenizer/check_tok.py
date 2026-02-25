
import os
import sys
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer.bpe import BPETokenizer

def check_tokenizer():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    tok_dir = config["tokenizer"]["save_dir"]
    tokenizer = BPETokenizer()
    
    if os.path.exists(tok_dir):
        print(f"Loading tokenizer from {tok_dir}...")
        tokenizer.load(tok_dir)
        
        text = "Hello world"
        ids = tokenizer.encode(text)
        tokens = [tokenizer._tokenizer.id_to_token(i) for i in ids]
        
        print(f"Text: '{text}'")
        print(f"IDs: {ids}")
        print(f"Tokens: {tokens}")
        print(f"Vocab size: {tokenizer.vocab_size}")
        print(f"SOS ID: {tokenizer.sos_id}")
        print(f"EOS ID: {tokenizer.eos_id}")
        print(f"PAD ID: {tokenizer.pad_id}")
        
        if tokens[0] == tokenizer.SOS_TOKEN and tokens[-1] == tokenizer.EOS_TOKEN:
            print("SUCCESS: Post-processor is working (SOS/EOS found).")
        else:
            print("FAILURE: Post-processor is NOT working (SOS/EOS missing).")
            
        decoded = tokenizer.decode(ids)
        print(f"Decoded: '{decoded}'")
    else:
        print(f"Tokenizer directory {tok_dir} not found.")

if __name__ == "__main__":
    check_tokenizer()
