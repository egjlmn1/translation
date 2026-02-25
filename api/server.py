"""FastAPI server — serves both web UIs and exposes translation + training WebSocket APIs."""
import json
import os
from typing import Any, Dict, Optional, Set

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.inference import TranslationInference
from model.transformer import TransformerModel
from tokenizer.base import BaseTokenizer
from training.metrics import MetricsTracker, compute_bleu


def create_app(
    config: Dict[str, Any],
    ws_clients: Optional[Set] = None,
    metrics: Optional[MetricsTracker] = None,
    tokenizer: Optional[BaseTokenizer] = None,
    model: Optional[TransformerModel] = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(title="Translation Platform")
    _ws_clients: Set = ws_clients if ws_clients is not None else set()
    _metrics = metrics or MetricsTracker()
    _inference: Optional[TranslationInference] = None

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    web_dir = os.path.join(project_root, "web")

    # Mount static assets
    app.mount("/static/translate", StaticFiles(directory=os.path.join(web_dir, "translate")), name="translate_static")
    app.mount("/static/train", StaticFiles(directory=os.path.join(web_dir, "train")), name="train_static")

    # ------------------------------------------------------------------
    # Lazy-load inference engine
    # ------------------------------------------------------------------

    def _get_inference() -> Optional[TranslationInference]:
        nonlocal _inference, model, tokenizer
        if _inference is not None:
            return _inference
        if model is not None and tokenizer is not None:
            _inference = TranslationInference(model, tokenizer)
            return _inference
        # Try loading from checkpoint
        try:
            from tokenizer.bpe import BPETokenizer
            tok_dir = os.path.join(project_root, config["tokenizer"]["save_dir"])
            if tokenizer is None:
                tokenizer = BPETokenizer()
                tokenizer.load(tok_dir)
            if model is None:
                mc = config["model"]
                model = TransformerModel(
                    src_vocab_size=tokenizer.vocab_size,
                    tgt_vocab_size=tokenizer.vocab_size,
                    d_model=mc["d_model"],
                    n_heads=mc["n_heads"],
                    n_encoder_layers=mc["n_encoder_layers"],
                    n_decoder_layers=mc["n_decoder_layers"],
                    d_ff=mc["d_ff"],
                    dropout=mc["dropout"],
                    max_seq_len=mc["max_seq_len"],
                    pad_id=tokenizer.pad_id,
                )
                ckpt_path = os.path.join(project_root, config["training"]["checkpoint_dir"], "checkpoint_latest.pt")
                if os.path.exists(ckpt_path):
                    model.load_checkpoint(ckpt_path)
                    print(f"[Server] Loaded model from {ckpt_path}")
                else:
                    print("[Server] WARNING: No checkpoint found — model has random weights.")
            _inference = TranslationInference(model, tokenizer)
            return _inference
        except Exception as e:
            print(f"[Server] Could not load model: {e}")
            return None

    # ------------------------------------------------------------------
    # Page routes
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return FileResponse(os.path.join(web_dir, "translate", "index.html"))

    @app.get("/translate", response_class=HTMLResponse)
    async def translate_page():
        return FileResponse(os.path.join(web_dir, "translate", "index.html"))

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_page():
        return FileResponse(os.path.join(web_dir, "train", "index.html"))

    # ------------------------------------------------------------------
    # Translation API
    # ------------------------------------------------------------------

    @app.post("/api/translate")
    async def translate_api(body: dict):
        text = body.get("text", "").strip()
        
        if not text:
            return JSONResponse({"error": "No text provided"}, status_code=400)

        results = {}
        
        # 0. Source Tokens
        inf = _get_inference()
        if inf is not None:
            results["src_tokens"] = tokenizer.tokenize(text)

        # 1. Google Translate prediction (Ground Truth)
        google_trans = ""
        try:
            from googletrans import Translator
            translator = Translator()
            google_result = translator.translate(text, src='en', dest='fr')
            google_trans = google_result.text
            results["google"] = {"translation": google_trans}
        except Exception as e:
            print(f"[Server] Google Translate error: {e}")
            results["google"] = {"error": str(e)}

        # 2. Model prediction
        inf = _get_inference()
        if inf is not None:
            model_result = inf.translate(text, max_len=config["model"]["max_seq_len"])
            model_trans = model_result.get("translation", "")
            
            # Calculate BLEU using Google as reference
            bleu_score = None
            if google_trans:
                bleu_score = round(compute_bleu(google_trans, model_trans) * 100, 2)
            
            # 3. Back-translation (Model FR -> Google EN)
            back_trans = ""
            try:
                from googletrans import Translator
                translator = Translator()
                back_result = translator.translate(model_trans, src='fr', dest='en')
                back_trans = back_result.text
            except Exception as e:
                print(f"[Server] Back-translate error: {e}")

            results["model"] = {
                "translation": model_trans,
                "bleu": bleu_score,
                "back_translation": back_trans
            }
        else:
            results["model"] = {"error": "Model not loaded"}

        return JSONResponse(results)

    # ------------------------------------------------------------------
    # Training status API
    # ------------------------------------------------------------------

    @app.get("/api/status")
    async def training_status():
        return JSONResponse(_metrics.to_dict())

    # ------------------------------------------------------------------
    # WebSocket for live training metrics
    # ------------------------------------------------------------------

    @app.websocket("/ws/training")
    async def training_ws(websocket: WebSocket):
        await websocket.accept()
        _ws_clients.add(websocket)
        # print(f"[Server] WebSocket client connected ({len(_ws_clients)} total)")
        try:
            # Send current state immediately
            await websocket.send_text(json.dumps(_metrics.to_dict()))
            # Keep connection alive
            while True:
                # Wait for any message (ping/pong or close)
                data = await websocket.receive_text()
                # Client can request a full refresh
                if data == "refresh":
                    await websocket.send_text(json.dumps(_metrics.to_dict()))
        except WebSocketDisconnect:
            pass
        finally:
            _ws_clients.discard(websocket)
            # print(f"[Server] WebSocket client disconnected ({len(_ws_clients)} total)")

    return app
