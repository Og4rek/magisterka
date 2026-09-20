from __future__ import annotations

import pytest
import torch

from pcam_repro.losses import (
    BinaryFocalLoss,
    GeneralizedCrossEntropy,
    NTXentLoss,
    SymmetricCrossEntropy,
)


@pytest.mark.parametrize(
    "criterion",
    [BinaryFocalLoss(), GeneralizedCrossEntropy(), SymmetricCrossEntropy()],
)
def test_binary_losses_have_finite_gradients(criterion: torch.nn.Module) -> None:
    logits = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 1.0])
    loss = criterion(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_nt_xent_has_finite_gradient() -> None:
    first = torch.randn(4, 16, requires_grad=True)
    second = torch.randn(4, 16, requires_grad=True)
    loss = NTXentLoss(0.2)(first, second)
    loss.backward()
    assert torch.isfinite(loss)
    assert first.grad is not None
