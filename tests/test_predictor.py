"""Comprehensive tests for predictor architecture components.

Covers shape correctness, causal masking, focal loss mathematics,
MPS device compatibility, seed determinism, and model consistency.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from lob_forge.predictor.deeplob import DeepLOB
from lob_forge.predictor.linear_baseline import LinearBaseline
from lob_forge.predictor.losses import FocalLoss
from lob_forge.predictor.model import DualAttentionTransformer
from lob_forge.predictor.spatial_attention import SpatialAttentionBlock
from lob_forge.predictor.temporal_attention import TemporalAttentionBlock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
B = 4
T = 20
D_MODEL = 64
N_LEVELS = 10
FEATURES_PER_LEVEL = 4
N_CLASSES = 3
N_HORIZONS = 4
INPUT_DIM = N_LEVELS * FEATURES_PER_LEVEL  # 40

MPS_AVAILABLE = torch.backends.mps.is_available()
MPS_SKIP = pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")


def _rand_input(batch: int = B, seq_len: int = T) -> torch.Tensor:
    """Random LOB-shaped input: (batch, T, 40)."""
    return torch.randn(batch, seq_len, INPUT_DIM)


# ---------------------------------------------------------------------------
# Spatial Attention
# ---------------------------------------------------------------------------
class TestSpatialAttentionBlock:
    def test_output_shape(self) -> None:
        block = SpatialAttentionBlock(d_model=D_MODEL)
        x = torch.randn(B * T, N_LEVELS, D_MODEL)
        out = block(x)
        assert out.shape == (B * T, N_LEVELS, D_MODEL)

    @pytest.mark.parametrize("batch", [1, 64])
    def test_different_batch_sizes(self, batch: int) -> None:
        block = SpatialAttentionBlock(d_model=D_MODEL)
        x = torch.randn(batch, N_LEVELS, D_MODEL)
        out = block(x)
        assert out.shape == (batch, N_LEVELS, D_MODEL)


# ---------------------------------------------------------------------------
# Temporal Attention
# ---------------------------------------------------------------------------
class TestTemporalAttentionBlock:
    def test_output_shape(self) -> None:
        block = TemporalAttentionBlock(d_model=D_MODEL)
        x = torch.randn(B, T, D_MODEL)
        out = block(x)
        assert out.shape == (B, T, D_MODEL)

    def test_causal_masking(self) -> None:
        """Modifying future inputs must not change past outputs."""
        block = TemporalAttentionBlock(d_model=D_MODEL)
        block.eval()

        x1 = torch.randn(1, T, D_MODEL)
        x2 = x1.clone()
        # Modify future time steps (index 10 onwards)
        x2[:, 10:, :] = torch.randn(1, T - 10, D_MODEL)

        with torch.no_grad():
            out1 = block(x1)
            out2 = block(x2)

        # Outputs at positions 0..9 must be identical
        assert torch.allclose(out1[:, :10, :], out2[:, :10, :], atol=1e-5)

    @pytest.mark.parametrize("seq_len", [50, 200])
    def test_variable_seq_len(self, seq_len: int) -> None:
        block = TemporalAttentionBlock(d_model=D_MODEL, max_seq_len=512)
        x = torch.randn(B, seq_len, D_MODEL)
        out = block(x)
        assert out.shape == (B, seq_len, D_MODEL)


# ---------------------------------------------------------------------------
# DualAttentionTransformer (TLOB)
# ---------------------------------------------------------------------------
class TestDualAttentionTransformer:
    def test_output_shapes(self) -> None:
        model = DualAttentionTransformer()
        x = _rand_input()
        out = model(x)
        assert out["logits"].shape == (B, N_HORIZONS, N_CLASSES)
        assert out["vpin"].shape == (B, 1)
        assert out["embedding"].shape == (B, D_MODEL)

    def test_without_vpin_head(self) -> None:
        model = DualAttentionTransformer(vpin_head=False)
        x = _rand_input()
        out = model(x)
        assert "vpin" not in out
        assert "logits" in out
        assert "embedding" in out

    @pytest.mark.parametrize("seq_len", [50, 100, 200])
    def test_different_seq_lengths(self, seq_len: int) -> None:
        model = DualAttentionTransformer()
        x = _rand_input(seq_len=seq_len)
        out = model(x)
        assert out["logits"].shape == (B, N_HORIZONS, N_CLASSES)

    def test_single_horizon(self) -> None:
        model = DualAttentionTransformer(n_horizons=1)
        x = _rand_input()
        out = model(x)
        assert out["logits"].shape == (B, 1, N_CLASSES)

    def test_deterministic_with_seed(self) -> None:
        model = DualAttentionTransformer()
        model.eval()
        x = _rand_input()

        torch.manual_seed(42)
        with torch.no_grad():
            out1 = model(x)

        torch.manual_seed(42)
        with torch.no_grad():
            out2 = model(x)

        assert torch.equal(out1["logits"], out2["logits"])
        assert torch.equal(out1["embedding"], out2["embedding"])


# ---------------------------------------------------------------------------
# DeepLOB
# ---------------------------------------------------------------------------
class TestDeepLOB:
    def test_output_shape(self) -> None:
        model = DeepLOB()
        x = _rand_input()
        out = model(x)
        assert out.shape == (B, N_HORIZONS, N_CLASSES)

    @pytest.mark.parametrize("seq_len", [50, 100])
    def test_different_seq_lengths(self, seq_len: int) -> None:
        model = DeepLOB()
        x = _rand_input(seq_len=seq_len)
        out = model(x)
        assert out.shape == (B, N_HORIZONS, N_CLASSES)

    def test_single_horizon(self) -> None:
        model = DeepLOB(n_horizons=1)
        x = _rand_input()
        out = model(x)
        assert out.shape == (B, 1, N_CLASSES)


# ---------------------------------------------------------------------------
# LinearBaseline
# ---------------------------------------------------------------------------
class TestLinearBaseline:
    def test_output_shape(self) -> None:
        model = LinearBaseline()
        x = _rand_input()
        out = model(x)
        assert out.shape == (B, N_HORIZONS, N_CLASSES)

    def test_uses_last_timestep(self) -> None:
        """Changing earlier time steps must not affect output."""
        model = LinearBaseline()
        model.eval()

        x1 = _rand_input()
        x2 = x1.clone()
        # Modify all time steps except the last
        x2[:, :-1, :] = torch.randn(B, T - 1, INPUT_DIM)

        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)

        assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------
class TestFocalLoss:
    def test_gamma_zero_matches_ce(self) -> None:
        logits = torch.randn(B, N_CLASSES)
        targets = torch.randint(0, N_CLASSES, (B,))

        fl = FocalLoss(gamma=0.0)
        fl_loss = fl(logits, targets)
        ce_loss = F.cross_entropy(logits, targets)

        assert torch.allclose(fl_loss, ce_loss, atol=1e-5)

    def test_focal_reduces_easy_loss(self) -> None:
        logits = torch.randn(B, N_CLASSES)
        targets = torch.randint(0, N_CLASSES, (B,))

        fl0 = FocalLoss(gamma=0.0)
        fl2 = FocalLoss(gamma=2.0)

        loss0 = fl0(logits, targets)
        loss2 = fl2(logits, targets)

        # Focal loss with gamma=2 should be <= gamma=0 (CE)
        assert loss2.item() <= loss0.item() + 1e-6

    def test_class_weights(self) -> None:
        logits = torch.randn(B, N_CLASSES)
        targets = torch.randint(0, N_CLASSES, (B,))
        weights = torch.tensor([1.0, 2.0, 3.0])

        fl_unweighted = FocalLoss(gamma=2.0)
        fl_weighted = FocalLoss(gamma=2.0, class_weights=weights)

        loss_uw = fl_unweighted(logits, targets)
        loss_w = fl_weighted(logits, targets)

        # Weighted loss should differ from unweighted
        assert not torch.allclose(loss_uw, loss_w)

    def test_3d_input(self) -> None:
        logits = torch.randn(B, N_HORIZONS, N_CLASSES)
        targets = torch.randint(0, N_CLASSES, (B, N_HORIZONS))

        fl = FocalLoss(gamma=2.0)
        loss = fl(logits, targets)

        assert loss.dim() == 0  # scalar

    def test_reduction_none(self) -> None:
        logits = torch.randn(B, N_HORIZONS, N_CLASSES)
        targets = torch.randint(0, N_CLASSES, (B, N_HORIZONS))

        fl = FocalLoss(gamma=2.0, reduction="none")
        loss = fl(logits, targets)

        assert loss.shape == (B, N_HORIZONS)

    def test_manual_calculation(self) -> None:
        """Hand-compute focal loss for 2 samples, verify match."""
        gamma = 2.0
        logits = torch.tensor([[2.0, 1.0, 0.5], [0.0, 3.0, 1.0]])
        targets = torch.tensor([0, 2])  # class 0, class 2

        # Manual computation
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather for true class
        pt_0 = probs[0, 0].item()
        pt_1 = probs[1, 2].item()
        log_pt_0 = log_probs[0, 0].item()
        log_pt_1 = log_probs[1, 2].item()

        fl_0 = -((1.0 - pt_0) ** gamma) * log_pt_0
        fl_1 = -((1.0 - pt_1) ** gamma) * log_pt_1
        expected_mean = (fl_0 + fl_1) / 2.0

        fl = FocalLoss(gamma=gamma)
        actual = fl(logits, targets)

        assert abs(actual.item() - expected_mean) < 1e-5


# ---------------------------------------------------------------------------
# Model Consistency
# ---------------------------------------------------------------------------
class TestModelConsistency:
    def test_all_models_same_output_shape(self) -> None:
        x = _rand_input(batch=B, seq_len=100)

        tlob = DualAttentionTransformer()
        deeplob = DeepLOB()
        linear = LinearBaseline()

        out_tlob = tlob(x)["logits"]
        out_deeplob = deeplob(x)
        out_linear = linear(x)

        expected = (B, N_HORIZONS, N_CLASSES)
        assert out_tlob.shape == expected
        assert out_deeplob.shape == expected
        assert out_linear.shape == expected


# ---------------------------------------------------------------------------
# MPS Compatibility
# ---------------------------------------------------------------------------
class TestMPSCompatibility:
    @MPS_SKIP
    def test_tlob_on_mps(self) -> None:
        device = torch.device("mps")
        model = DualAttentionTransformer().to(device)
        x = _rand_input().to(device)
        out = model(x)
        assert out["logits"].device.type == "mps"

    @MPS_SKIP
    def test_deeplob_on_mps(self) -> None:
        device = torch.device("mps")
        model = DeepLOB().to(device)
        x = _rand_input().to(device)
        out = model(x)
        assert out.device.type == "mps"

    @MPS_SKIP
    def test_linear_on_mps(self) -> None:
        device = torch.device("mps")
        model = LinearBaseline().to(device)
        x = _rand_input().to(device)
        out = model(x)
        assert out.device.type == "mps"

    @MPS_SKIP
    def test_focal_loss_on_mps(self) -> None:
        device = torch.device("mps")
        fl = FocalLoss(gamma=2.0).to(device)
        logits = torch.randn(B, N_CLASSES).to(device)
        targets = torch.randint(0, N_CLASSES, (B,)).to(device)
        loss = fl(logits, targets)
        assert loss.device.type == "mps"
