import math
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional
import fractions

# Fix for NLTK BLEU score bug in Python 3.12+ 
# (TypeError: Fraction.__new__() got an unexpected keyword argument '_normalize')
_original_fraction_new = fractions.Fraction.__new__
def _fixed_fraction_new(cls, numerator=0, denominator=None, *, _normalize=True):
    return _original_fraction_new(cls, numerator, denominator)
fractions.Fraction.__new__ = _fixed_fraction_new

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def compute_bleu(reference: str, hypothesis: str) -> float:
    """
    Compute a simple sentence-level BLEU score using NLTK.
    Uses unigram weights for single-word inputs, and default weights for full sentences.
    """
    smoothie = SmoothingFunction().method1
    ref_tokens = reference.strip().split()
    hyp_tokens = hypothesis.strip().split()
    
    # If it's a single word, use only unigrams to avoid 0.0 scores from missing higher n-grams
    weights = (1.0, 0, 0, 0) if len(ref_tokens) == 1 and len(hyp_tokens) == 1 else (0.25, 0.25, 0.25, 0.25)
    
    # NLTK expects a list of references, each being a list of tokens
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie, weights=weights)


class MetricsTracker:
    """Accumulates training metrics and provides history for the dashboard."""

    def __init__(self):
        self.train_losses: List[Dict] = []   # [{step, loss}]
        self.val_losses: List[Dict] = []     # [{step, loss}]
        self.learning_rates: List[Dict] = [] # [{step, lr}]
        self.bleu_scores: List[Dict] = []    # [{step, bleu}]
        self.speed_history: List[Dict] = []  # [{step, speed}]
        self.sample_translations: List[Dict] = []  # [{step, src, ref, pred}]
        self.epoch: int = 0
        self.total_epochs: int = 0
        self.global_step: int = 0
        self.total_steps_estimate: int = 0
        self.training_active: bool = False
        self.start_time: Optional[float] = None
        self.samples_per_sec: float = 0.0
        self._recent_step_time: float = 0.0
        self._speed_window: deque = deque()  # (timestamp, samples)

    def start_training(self, total_steps_estimate: int = 0) -> None:
        self.training_active = True
        self.start_time = time.time()
        self.total_steps_estimate = total_steps_estimate
        self.reset_timer()

    def reset_timer(self) -> None:
        self._recent_step_time = time.time()
        self._speed_window.clear()
        self.samples_per_sec = 0.0

    def stop_training(self) -> None:
        self.training_active = False

    def log_train_loss(self, step: int, loss: float) -> None:
        self.global_step = step
        self.train_losses.append({"step": step, "loss": round(loss, 6)})

    def log_val_loss(self, step: int, loss: float) -> None:
        self.val_losses.append({"step": step, "loss": round(loss, 6)})

    def log_lr(self, step: int, lr: float) -> None:
        self.learning_rates.append({"step": step, "lr": lr})

    def log_bleu(self, step: int, bleu: float) -> None:
        self.bleu_scores.append({"step": step, "bleu": round(bleu, 4)})

    def log_speed(self, step: int) -> None:
        """Log current 30s average speed to history."""
        self.speed_history.append({"step": step, "speed": round(self.samples_per_sec, 1)})
        if len(self.speed_history) > 1000:
            self.speed_history = self.speed_history[-1000:]

    def log_sample(self, step: int, src: str, ref: str, pred: str) -> None:
        self.sample_translations.append({
            "step": step,
            "src": src,
            "ref": ref,
            "pred": pred,
        })
        # Keep only last 20 samples
        if len(self.sample_translations) > 20:
            self.sample_translations = self.sample_translations[-20:]

    def update_speed(self, batch_size: int) -> None:
        """Add current batch to sliding window and update samples_per_sec (avg over 30s)."""
        now = time.time()
        self._speed_window.append((now, batch_size))
        
        # Keep only last 30 seconds
        cutoff = now - 30.0
        while self._speed_window and self._speed_window[0][0] < cutoff:
            self._speed_window.popleft()
            
        if len(self._speed_window) > 1:
            total_samples = sum(s[1] for s in self._speed_window)
            elapsed = now - self._speed_window[0][0]
            if elapsed > 0:
                self.samples_per_sec = total_samples / elapsed
        elif self._recent_step_time > 0:
            elapsed = now - self._recent_step_time
            if elapsed > 0:
                self.samples_per_sec = batch_size / elapsed
        
        self._recent_step_time = now

    @property
    def eta_seconds(self) -> float:
        if not self.training_active or not self.start_time or self.global_step == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        steps_remaining = max(self.total_steps_estimate - self.global_step, 0)
        secs_per_step = elapsed / self.global_step
        return steps_remaining * secs_per_step

    def to_dict(self) -> Dict:
        """Serialize full state for the dashboard WebSocket, handling NaN safely."""
        def clean_val(v):
            if isinstance(v, (float, int)) and (math.isnan(v) or math.isinf(v)):
                return None
            return v

        return {
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "global_step": self.global_step,
            "total_steps_estimate": self.total_steps_estimate,
            "training_active": self.training_active,
            "eta_seconds": clean_val(round(self.eta_seconds, 1)),
            "samples_per_sec": clean_val(round(self.samples_per_sec, 1)),
            "train_losses": [{"step": d["step"], "loss": clean_val(d["loss"])} for d in self.train_losses],
            "val_losses": [{"step": d["step"], "loss": clean_val(d["loss"])} for d in self.val_losses],
            "learning_rates": self.learning_rates,
            "bleu_scores": [{"step": d["step"], "bleu": clean_val(d["bleu"])} for d in self.bleu_scores],
            "speed_history": [{"step": d["step"], "speed": clean_val(d["speed"])} for d in self.speed_history],
            "sample_translations": self.sample_translations,
        }
