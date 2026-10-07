"""Main-model-only sample clock; baseline optimization remains unchanged."""

from __future__ import annotations

import math

from stock_forecasting.optimization_policy import ValidationPlateauScheduler


class SamplePlateauScheduler(ValidationPlateauScheduler):
    """Combine exposure-based cosine decay with validation-driven reductions.

    Counts actual consumed windows, not nominal epochs or hardware-dependent
    optimizer steps. Resume realignment must never reset this sample clock.
    """

    def __init__(
        self, optimizer, warmup_steps, *, warmup_samples, decay_start, decay_end, **kwargs
    ):
        if not 0 < warmup_samples <= decay_start < decay_end:
            raise ValueError("Sample schedule requires 0 < warmup <= start < end")
        self.processed_samples = 0
        self.previous_validation_samples = 0
        self.previous_validation_at_floor = False
        self.warmup_samples = int(warmup_samples)
        self.decay_start, self.decay_end = int(decay_start), int(decay_end)
        super().__init__(optimizer, warmup_steps, **kwargs)

    @property
    def sample_ratio(self):
        progress = min(1.0, max(0.0, (self.processed_samples - self.decay_start)
                                / (self.decay_end - self.decay_start)))
        return self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    def _apply(self):
        warmup = min(1.0, max(1, self.processed_samples) / self.warmup_samples)
        ratio = min(self.ratio, self.sample_ratio) * warmup
        for group, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base * ratio

    def step(self, samples=None):
        if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
            raise ValueError("Sample schedule requires the actual positive consumed window count")
        self.processed_samples += samples
        self.last_epoch += 1
        self._apply()

    def observe(self, value, min_delta=0.0):
        if not math.isfinite(value):
            raise ValueError("Sample scheduler requires a finite validation loss")
        warmed_up = self.processed_samples >= self.warmup_samples
        # Reaching the floor inside this interval is not an entire interval at
        # the floor. Repeated validation without training cannot satisfy it.
        if (warmed_up and self.previous_validation_at_floor
                and self.processed_samples > self.previous_validation_samples):
            self.low_evaluations += 1
        if self.best is None or value < self.best - min_delta:
            self.best, self.stale = value, 0
        elif warmed_up:
            self.stale += 1
            if self.stale >= self.patience:
                self.ratio = max(self.min_ratio, self.ratio * self.factor)
                self.stale = 0
                self._apply()
        self.previous_validation_samples = self.processed_samples
        self.previous_validation_at_floor = (
            warmed_up and min(self.ratio, self.sample_ratio) <= self.min_ratio * (1 + 1e-8)
        )
