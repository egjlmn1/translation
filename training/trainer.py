"""Training loop with checkpointing, LR scheduling, and live metrics push."""
import asyncio
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Set

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.base import BaseModel
from tokenizer.base import BaseTokenizer
from training.metrics import MetricsTracker, compute_bleu


class Trainer:
    """
    Handles the full training loop:
      - Cross-entropy loss with label smoothing
      - Adam optimizer + warmup linear / inverse-sqrt LR schedule
      - Gradient clipping
      - Checkpoint save/resume
      - Pushes metrics to connected WebSocket clients
    """

    def __init__(
        self,
        model: BaseModel,
        tokenizer: BaseTokenizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
        metrics: MetricsTracker,
        ws_clients: Optional[Set] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.metrics = metrics
        self.ws_clients: Set = ws_clients or set()

        # Training config
        tc = config["training"]
        self.epochs = tc["epochs"]
        self.lr = tc["learning_rate"]
        self.warmup_steps = tc["warmup_steps"]
        self.grad_clip = tc["grad_clip"]
        self.label_smoothing = tc["label_smoothing"]
        self.checkpoint_dir = tc["checkpoint_dir"]
        self.checkpoint_interval = tc["checkpoint_interval"]
        self.eval_interval = tc["eval_interval"]
        self.log_interval = tc["log_interval"]
        self.batch_size = tc["batch_size"]

        # Set total units in metrics
        self.metrics.total_epochs = self.epochs
        # BF16 disabled — causes random CUBLAS_STATUS_EXECUTION_FAILED on some GPUs.
        # FP16 uses a more stable CUDA code path.
        self.use_bf16 = True
        
        # Estimate total steps for ETA calculation
        try:
            num_batches = len(self.train_loader)
            self.total_steps_estimate = self.epochs * num_batches
            self.metrics.total_steps_estimate = self.total_steps_estimate
            
            # Auto-adjust intervals for small datasets so we actually see updates
            if num_batches < self.log_interval:
                self.log_interval = max(1, num_batches // 2)
            if num_batches < self.eval_interval:
                self.eval_interval = max(1, num_batches)
            
            print(f"[Trainer] Estimated total steps: {self.total_steps_estimate}")
            print(f"[Trainer] Intervals: log={self.log_interval}, eval={self.eval_interval}")
        except (TypeError, AttributeError):
            # For IterableDataset where len() might not be known
            self.total_steps_estimate = 0


        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Loss
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_id,
            label_smoothing=self.label_smoothing,
        )

        # Optimizer - Use 'fused=True' for massive speedup on GPU
        use_fused = torch.cuda.is_available() and "cuda" in str(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01, # Crucial for Transformers
            fused=use_fused
        )

        # State
        self.global_step = 0
        self.start_epoch = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    # LR schedule  (warmup then inverse sqrt)
    # ------------------------------------------------------------------

    def _get_lr(self, step: int) -> float:
        if step == 0:
            step = 1
        warmup = self.warmup_steps
        # Standard Noam normalization factor
        scale = (warmup ** 0.5) * self.lr 
        return scale * min(step ** -0.5, step * warmup ** -1.5)

    def _update_lr(self) -> float:
        lr = self._get_lr(self.global_step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    # ------------------------------------------------------------------
    # Checkpoint save / resume
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: str = "latest") -> str:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.pt")
        self.model.save_checkpoint(path, extra={
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.metrics.epoch,
            "train_losses": self.metrics.train_losses[-1000:],  # keep last 1k
            "val_losses": self.metrics.val_losses,
            "bleu_scores": self.metrics.bleu_scores,
            "sample_translations": self.metrics.sample_translations,
            "learning_rates": self.metrics.learning_rates[-1000:],
        })
        print(f"[Trainer] Saved checkpoint to {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        checkpoint = self.model.load_checkpoint(path, device=self.device)
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.start_epoch = checkpoint.get("epoch", 0)
        self.metrics.global_step = self.global_step
        self.metrics.epoch = self.start_epoch
        # Restore metric history
        self.metrics.train_losses = checkpoint.get("train_losses", [])
        self.metrics.val_losses = checkpoint.get("val_losses", [])
        self.metrics.bleu_scores = checkpoint.get("bleu_scores", [])
        self.metrics.sample_translations = checkpoint.get("sample_translations", [])
        self.metrics.learning_rates = checkpoint.get("learning_rates", [])
        print(f"[Trainer] Resumed from {path} (step={self.global_step}, epoch={self.start_epoch})")
        # Reset timer after load
        self.metrics.reset_timer()

    # ------------------------------------------------------------------
    # WebSocket push
    # ------------------------------------------------------------------

    def _push_metrics(self) -> None:
        """Send current metrics to all connected WebSocket clients."""
        if not self.ws_clients:
            return
        try:
            data = json.dumps(self.metrics.to_dict())
        except Exception as e:
            # If cleaning failed, don't crash the whole training loop
            return
        dead: Set = set()

        # Try to get or create event loop for async sends
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        for ws in self.ws_clients:
            try:
                asyncio.run_coroutine_threadsafe(ws.send_text(data), loop)
            except Exception:
                dead.add(ws)
        self.ws_clients -= dead

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation on the validation set. Returns average loss."""
        if self.val_loader is None:
            return 0.0
        self.model.eval()
        total_loss = 0.0
        count = 0
        for src, tgt in self.val_loader:
            src, tgt = src.to(self.device), tgt.to(self.device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # SAFETY: Skip empty sequences to prevent CUDA crash
            if tgt_input.size(1) == 0:
                continue

            with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16 if self.use_bf16 else torch.float32):
                logits = self.model(src, tgt_input)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_output.reshape(-1),
                )
            
            total_loss += loss.item()
            count += 1
            if count >= 50:  # cap eval batches for speed
                break
        self.model.train()
        return total_loss / max(count, 1)

    @torch.no_grad()
    def generate_sample(self) -> Optional[Dict[str, str]]:
        """Generate a sample translation from the validation set."""
        if self.val_loader is None:
            return None
        self.model.eval()
        
        with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16 if self.use_bf16 else torch.float32):
            for src, tgt in self.val_loader:
                # Try a few samples from this batch
                for i in range(min(len(src), 5)):
                    src_sent = src[i:i+1].to(self.device)
                    
                    # SAFETY: If source sequence length is 0, greedy_decode might crash or produce empty output
                    if src_sent.size(1) == 0:
                        print(f"[Trainer] WARNING: Found empty source input in generate_sample. Skipping sample.")
                        continue

                    ref_ids = tgt[i].tolist()
                    
                    output_ids = self.model.greedy_decode(
                        src_sent,
                        sos_id=self.tokenizer.sos_id,
                        eos_id=self.tokenizer.eos_id,
                        max_len=self.config["model"]["max_seq_len"],
                    )
                    
                    src_text = self.tokenizer.decode(src[i].tolist())
                    ref_text = self.tokenizer.decode(ref_ids)
                    pred_text = self.tokenizer.decode(output_ids[0].tolist())
                    
                    # If we found a non-empty prediction, or this is the first sample, return it
                    if pred_text.strip() or i == 0:
                        if not pred_text.strip():
                             print(f"[Trainer] Sample {i} source: '{src_text}' -> Empty prediction (IDs: {output_ids[0].tolist()})")
                        
                        self.model.train()
                        return {"src": src_text, "ref": ref_text, "pred": pred_text}
                
                # If we tried 5 samples and nothing, just break and return None
                break
                
        self.model.train()
        return None

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop."""
        self.model.train()
        self.metrics.start_training(total_steps_estimate=self.total_steps_estimate)

        print(f"[Trainer] Starting training on {self.device}")
        print(f"[Trainer] Epochs: {self.epochs}, Batch size: {self.batch_size}")
        print(f"[Trainer] Warmup steps: {self.warmup_steps}")

        # Ensure starting LR is correct (especially after loading checkpoint)
        lr = self._update_lr()

        for epoch in range(self.start_epoch, self.epochs):
            self.metrics.epoch = epoch
            epoch_loss = 0.0
            epoch_steps = 0
            window_loss = 0.0
            window_steps = 0

            total_batches = None
            try:
                total_batches = len(self.train_loader)
            except (TypeError, AttributeError):
                pass
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch", total=total_batches)
            
            # Profiling logic
            time_data = 0.0
            time_forward = 0.0
            time_backward = 0.0
            time_other = 0.0
            # last_print_time = time.time()
            
            # Start timer before iterating to measure data loading
            # t0 = time.time()
            
            # TRAINING LOOP
            for src, tgt in pbar:
                # Keep CPU copies for crash debugging (free — DataLoader yields CPU tensors)
                src_cpu = src
                tgt_cpu = tgt

                # 1. Moving to device
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                # Teacher forcing: input is tgt[:-1], label is tgt[1:]
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # SAFETY: If seq length is 0, CUDA attention kernels will crash with 'illegal instruction'
                if tgt_input.size(1) == 0:
                    print(f"[Trainer] WARNING: Found empty target input at step {self.global_step}. Skipping batch to prevent CUDA crash.")
                    continue

                # 3. Forward pass with Mixed Precision
                try:
                    with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16 if self.use_bf16 else torch.float32):
                        logits = self.model(src, tgt_input)
                            
                        loss = self.criterion(
                            logits.reshape(-1, logits.size(-1)),
                            tgt_output.reshape(-1),
                        )

                    # SAFETY CHECK: Skip batch if loss is NaN/Inf to prevent CUDA illegal instruction
                    if not torch.isfinite(loss):
                        print(f"[Trainer] Warning: Loss is {loss.item()} at step {self.global_step}. Skipping batch.")
                        continue

                    # 4. Backward & Step
                    self.optimizer.zero_grad()
                    
                    loss.backward()
                    total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    # GRADIENT SAFETY CHECK:
                    # If the norm is NaN or Inf, the gradients are broken. 
                    # We skip the optimizer.step() to protect the model weights.
                    if torch.isfinite(total_norm):
                        self.optimizer.step()
                    else:
                        print(f"[Trainer] Warning: Gradient norm is {total_norm.item()}. Skipping step {self.global_step}.")

                except RuntimeError as e:
                    if "CUDA" in str(e) or "illegal" in str(e) or "launch" in str(e):
                        # === CRASH DUMP using CPU copies (GPU context is dead) ===
                        crash_path = os.path.join(self.checkpoint_dir, "debug_crash_batch.pt")
                        print(f"\n{'='*60}")
                        print(f"[CRASH DEBUG] CUDA error at step {self.global_step}")
                        print(f"  Error: {e}")
                        print(f"  src shape:       {src_cpu.shape}, dtype: {src_cpu.dtype}")
                        print(f"  tgt shape:       {tgt_cpu.shape}, dtype: {tgt_cpu.dtype}")
                        print(f"  src range:       [{src_cpu.min().item()}, {src_cpu.max().item()}]")
                        print(f"  tgt range:       [{tgt_cpu.min().item()}, {tgt_cpu.max().item()}]")
                        print(f"  vocab_size:      {self.model.src_embedding.num_embeddings}")
                        print(str(e))
                        
                        # Check for OOB tokens
                        vocab = self.model.src_embedding.num_embeddings
                        oob_src = (src_cpu >= vocab) | (src_cpu < 0)
                        oob_tgt = (tgt_cpu >= vocab) | (tgt_cpu < 0)
                        if oob_src.any():
                            print(f"  ⚠ OOB src tokens: {src_cpu[oob_src].tolist()}")
                        if oob_tgt.any():
                            print(f"  ⚠ OOB tgt tokens: {tgt_cpu[oob_tgt].tolist()}")
                        if not oob_src.any() and not oob_tgt.any():
                            print(f"  All token IDs in bounds: YES")
                        
                        torch.save({
                            "src": src_cpu,
                            "tgt": tgt_cpu,
                            "global_step": self.global_step,
                            "epoch": epoch,
                        }, crash_path)
                        print(f"  Batch saved to: {crash_path}")
                        print(f"  Run: python scripts/replay_crash.py")
                        print(f"{'='*60}\n")
                    # Save checkpoint before crashing so no progress is lost
                    try:
                        self.save_checkpoint(tag="latest")

                        print("[CRASH DEBUG] Saved checkpoint before exit.")
                    except Exception:
                        pass  # CUDA is dead, saving may fail
                    raise  # Re-raise so training stops (use train_with_restart.py to auto-resume)

                # --- Periodic model health check (every 50 steps, cheap) ---
                # if self.global_step % 50 == 0:
                #     for pname, p in self.model.named_parameters():
                #         if torch.isnan(p).any() or torch.isinf(p).any():
                #             print(f"[HEALTH CHECK] Step {self.global_step}: CORRUPTED weight '{pname}' "
                #                   f"(nan={torch.isnan(p).sum().item()}, inf={torch.isinf(p).sum().item()}, "
                #                   f"range=[{p.min().item():.4f}, {p.max().item():.4f}])")

                
                # Update LR
                lr = self._update_lr()
                
                # if torch.cuda.is_available(): torch.cuda.synchronize()
                # t3 = time.time()
                # time_backward += t3 - t2


                # Metrics
                loss_val = loss.item()
                epoch_loss += loss_val
                epoch_steps += 1
                window_loss += loss_val
                window_steps += 1
                self.metrics.global_step = self.global_step
                self.metrics.update_speed(src.size(0))

                # Log to dashboard
                if self.global_step % self.log_interval == 0:
                    avg_loss = window_loss / max(window_steps, 1)
                    self.metrics.log_train_loss(self.global_step, avg_loss)
                    self.metrics.log_lr(self.global_step, lr)
                    self.metrics.log_speed(self.global_step)
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                    window_loss = 0.0
                    window_steps = 0
                    self._push_metrics()

                # Increment global step at the very end of the batch processing
                self.global_step += 1
                
                # Check eval/checkpoint milestones after the increment
                if self.global_step % self.eval_interval == 0:
                    val_loss = self.evaluate()
                    self.metrics.log_val_loss(self.global_step, val_loss)
                    self._push_metrics()

                    sample = self.generate_sample()
                    if sample:
                        self.metrics.log_sample(
                            self.global_step,
                            sample["src"],
                            sample["ref"],
                            sample["pred"],
                        )
                        bleu = compute_bleu(sample["ref"], sample["pred"])
                        self.metrics.log_bleu(self.global_step, bleu)
                        self._push_metrics()
                    
                    # Reset timer after potential long evaluation to keep speed stats accurate
                    self.metrics.reset_timer()
                    t_start = time.perf_counter()
                else:
                    # Normal batch end, reset t_start for the next 'data' loading phase
                    t_start = time.perf_counter()

                # Checkpoint
                if self.global_step % self.checkpoint_interval == 0:
                    self.save_checkpoint(tag="latest")
                    self.save_checkpoint(tag=f"step_{self.global_step}")
                    self.metrics.reset_timer() # Don't count disk IO in training speed
                    t_start = time.perf_counter()

                # Prepare for next iteration's data loading timer
                # if torch.cuda.is_available(): torch.cuda.synchronize()
                # t4 = time.time()
                # time_other += t4 - t3
                
                # if t4 - last_print_time >= 10.0:
                #     total_time = time_data + time_forward + time_backward + time_other
                #     # if total_time > 0:
                #     #     print(f"\n[Profiling] Data: {time_data/total_time:.1%} | Fwd: {time_forward/total_time:.1%} | Bwd: {time_backward/total_time:.1%} | Other: {time_other/total_time:.1%}")
                #     time_data = time_forward = time_backward = time_other = 0.0
                #     last_print_time = t4
                    
                # t0 = time.time()

            print(f"[Epoch {epoch + 1}/{self.epochs}] avg_loss={epoch_loss / max(epoch_steps, 1):.4f}")
            self.save_checkpoint(tag="latest")

        self.metrics.stop_training()
        self.save_checkpoint(tag="final")
        print("[Trainer] Training complete!")
        self._push_metrics()
