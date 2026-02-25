"""Train the translation model."""
import argparse
import asyncio
import os
import sys
import threading
import warnings
from typing import Set

import yaml
import torch
import warnings

# Suppress prototype/internal warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Support for mismatched key_padding_mask and attn_mask.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*The PyTorch API of nested tensors is in prototype stage.*")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import create_dataloaders
from model.transformer import TransformerModel
from tokenizer.bpe import BPETokenizer
from training.metrics import MetricsTracker
from training.trainer import Trainer


# Shared state for WebSocket clients (will be populated by server if running)
ws_clients: Set = set()
metrics = MetricsTracker()


def main():
    parser = argparse.ArgumentParser(description="Train translation model")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--with-server", action="store_true", help="Launch dashboard server alongside training")
    args = parser.parse_args()

    # Set matmul precision for TensorCores (RTX 30/40 series)
    # torch.set_float32_matmul_precision('high')
    # torch.autograd.set_detect_anomaly(True)
    
    # Load config
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load tokenizer
    tok_type = config["tokenizer"].get("type", "bpe")
    if tok_type == "pretrained":
        from tokenizer.pretrained import PretrainedTokenizer
        tokenizer = PretrainedTokenizer(config["tokenizer"].get("model_name", "xlm-roberta-base"))
    else:
        from tokenizer.bpe import BPETokenizer
        tokenizer = BPETokenizer()
    
    tok_dir = os.path.join(project_root, config["tokenizer"]["save_dir"])
    if os.path.exists(tok_dir):
        tokenizer.load(tok_dir)
    elif tok_type == "bpe":
        print(f"[train_model] WARNING: Tokenizer directory {tok_dir} not found. Ensure it is trained.")

    # Create data loaders
    csv_path = os.path.join(project_root, config["data"]["csv_path"])
    train_loader, val_loader = create_dataloaders(  
        csv_path=csv_path,
        tokenizer=tokenizer,
        batch_size=config["training"]["batch_size"],
        max_seq_len=config["data"]["max_seq_len"],
        chunk_size=config["data"]["chunk_size"],
        shuffle_buffer=config["data"]["shuffle_buffer"],
        val_split=config["data"]["val_split"],
        max_rows=config["data"]["max_rows"],
        num_workers=config["training"].get("num_workers", 0),
        src_col=config["data"]["src_col"],
        tgt_col=config["data"]["tgt_col"],
        use_bucketing=config["data"].get("use_bucketing", False),
    )
    print(f"[train_model] Train loader: {len(train_loader)} batches")
    print(f"[train_model] Val loader: {len(val_loader)} batches")
    # Create model
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

    # model = torch.compile(model, mode="reduce-overhead")

    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        metrics=metrics,
        ws_clients=ws_clients,
    )

    # Resume if requested
    if args.resume:
        resume_path = os.path.join(project_root, args.resume)
        trainer.load_checkpoint(resume_path)

    # Optionally start the dashboard server in a background thread
    if args.with_server:
        server_thread = threading.Thread(
            target=_run_server,
            args=(config, ws_clients, metrics, tokenizer, model),
            daemon=True,
        )
        server_thread.start()
        print(f"[train_model] Dashboard available at http://localhost:{config['server']['port']}/dashboard")

    # Train
    trainer.train()


def _run_server(config, ws_clients_ref, metrics_ref, tokenizer, model):
    """Run the FastAPI server in a background thread."""
    from api.server import create_app
    import uvicorn

    app = create_app(
        config=config,
        ws_clients=ws_clients_ref,
        metrics=metrics_ref,
        tokenizer=tokenizer,
        model=model,
    )
    uvicorn.run(
        app,
        host=config["server"]["host"],
        port=config["server"]["port"],
        log_level="warning",
    )


if __name__ == "__main__":
    main()
