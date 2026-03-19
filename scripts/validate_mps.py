#!/usr/bin/env python
"""Validate PyTorch MPS backend for Apple Silicon training.

Runs a small forward+backward pass with float32 on the best available
device (MPS > CUDA > CPU) and prints a timing summary.
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn as nn


class SmallMLP(nn.Module):
    """Two-layer MLP for device validation (~1 000 params)."""

    def __init__(
        self, in_features: int = 40, hidden: int = 20, out_features: int = 1
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def detect_device() -> torch.device:
    """Return the best available device, preferring MPS."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def validate(device: torch.device) -> None:
    """Run a forward+backward pass on *device* and report timing."""
    print(f"Device : {device}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # ---------- setup ----------
    model = SmallMLP().to(device=device, dtype=torch.float32)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    x = torch.randn(16, 40, device=device, dtype=torch.float32)
    target = torch.randn(16, 1, device=device, dtype=torch.float32)

    # ---------- forward ----------
    t0 = time.perf_counter()
    output = model(x)
    if device.type == "mps":
        torch.mps.synchronize()
    t_forward = time.perf_counter() - t0

    # ---------- backward ----------
    loss = criterion(output, target)
    t1 = time.perf_counter()
    loss.backward()
    if device.type == "mps":
        torch.mps.synchronize()
    t_backward = time.perf_counter() - t1

    # ---------- optimizer step ----------
    t2 = time.perf_counter()
    optimizer.step()
    if device.type == "mps":
        torch.mps.synchronize()
    t_optim = time.perf_counter() - t2

    # ---------- checks ----------
    assert (
        output.device.type == device.type
    ), f"output on {output.device}, expected {device}"
    assert not torch.isnan(output).any(), "NaN in output"
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"

    # ---------- summary ----------
    print()
    print("Timing")
    print(f"  forward : {t_forward * 1000:7.2f} ms")
    print(f"  backward: {t_backward * 1000:7.2f} ms")
    print(f"  optim   : {t_optim * 1000:7.2f} ms")
    print(f"  total   : {(t_forward + t_backward + t_optim) * 1000:7.2f} ms")

    if device.type == "mps":
        try:
            allocated = torch.mps.current_allocated_memory()
            print(f"  MPS mem : {allocated / 1024:.1f} KB")
        except AttributeError:
            pass  # API not available in this PyTorch version

    print()
    print("PASS")


def main() -> None:
    """Entry point."""
    device = detect_device()
    if device.type == "cpu":
        print("WARNING: MPS not available, falling back to CPU")
        print()
    try:
        validate(device)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
