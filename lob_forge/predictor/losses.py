"""Focal loss for class-imbalanced LOB label prediction.

Focal loss (Lin et al., 2017) down-weights well-classified examples so the
model focuses on hard, misclassified cases.  This is critical for LOB mid-price
movement prediction where the STATIONARY class dominates.

When ``gamma=0`` and no class weights are provided, focal loss degrades to
standard cross-entropy.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class FocalLoss(nn.Module):
    """Focal loss with optional per-class weights.

    Parameters
    ----------
    gamma : float
        Focusing parameter.  Higher values increase focus on hard examples.
        ``gamma=0`` recovers standard cross-entropy.
    class_weights : Tensor | None
        Per-class weights for handling class imbalance (e.g., inverse
        frequency).  Registered as a buffer so it moves with ``.to(device)``.
    reduction : str
        ``"mean"``, ``"sum"``, or ``"none"``.

    Forward signature
    -----------------
    logits : Tensor
        ``(batch, n_classes)`` or ``(batch, n_horizons, n_classes)``.
    targets : Tensor
        ``(batch,)`` or ``(batch, n_horizons)`` of dtype ``int64``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights: Tensor | None = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss.

        Parameters
        ----------
        logits : Tensor
            ``(batch, n_classes)`` or ``(batch, n_horizons, n_classes)``.
        targets : Tensor
            ``(batch,)`` or ``(batch, n_horizons)`` of dtype ``int64``.

        Returns
        -------
        Tensor
            Scalar (if reduction is ``"mean"`` or ``"sum"``) or per-element
            losses (if ``"none"``).
        """
        # Handle 3D logits: (batch, n_horizons, n_classes)
        reshaped = False
        if logits.dim() == 3:
            B, H, C = logits.shape
            logits = logits.reshape(B * H, C)
            targets = targets.reshape(B * H)
            reshaped = True

        # 1. Log-softmax
        log_p = F.log_softmax(logits, dim=-1)

        # 2. Gather log-prob for true class
        log_pt = log_p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # 3. Compute pt
        pt = log_pt.exp()

        # 4. Focal weight
        focal_weight = (1.0 - pt) ** self.gamma

        # 5. Apply class weights if provided
        if self.class_weights is not None:
            w = self.class_weights[targets]
            focal_weight = focal_weight * w

        # 6. Loss
        loss = -focal_weight * log_pt

        # Reshape back if needed before reduction
        if reshaped and self.reduction == "none":
            loss = loss.reshape(B, H)

        # 7. Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
