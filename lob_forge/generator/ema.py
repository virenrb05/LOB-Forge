"""Exponential Moving Average wrapper for stable inference weights."""

from __future__ import annotations

import torch
import torch.nn as nn


class ExponentialMovingAverage:
    """Maintain an exponential moving average of model parameters.

    Shadow parameters track a smoothed version of the training weights,
    providing more stable predictions at inference time.

    Parameters
    ----------
    model : nn.Module
        Model whose parameters to track.
    decay : float
        EMA decay factor (typically 0.9999).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow_params: dict[str, torch.Tensor] = {
            name: p.data.detach().clone() for name, p in model.named_parameters()
        }
        self._backup: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters toward current model parameters.

        For each parameter: ``shadow = decay * shadow + (1 - decay) * param``.
        """
        for name, p in model.named_parameters():
            self.shadow_params[name].mul_(self.decay).add_(
                p.data, alpha=1.0 - self.decay
            )

    def apply_shadow(self, model: nn.Module) -> None:
        """Copy shadow parameters into model (for inference).

        Original parameters are saved internally so they can be restored
        with :meth:`restore`.
        """
        self._backup = {
            name: p.data.detach().clone() for name, p in model.named_parameters()
        }
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow_params[name])

    def restore(self, model: nn.Module) -> None:
        """Restore original parameters after inference."""
        for name, p in model.named_parameters():
            p.data.copy_(self._backup[name])
        self._backup = {}

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return shadow parameters for checkpoint saving."""
        return dict(self.shadow_params)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load shadow parameters from a checkpoint."""
        self.shadow_params = {k: v.detach().clone() for k, v in state_dict.items()}
