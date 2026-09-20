from __future__ import annotations

import pytest
import torch
from torch import nn

from pcam_repro.config import ModelConfig
from pcam_repro.models import build_model
from pcam_repro.models.cnn import InceptionPCam
from pcam_repro.models.d4 import (
    D4GroupConv2d,
    D4LiftingConv2d,
    d4_inverse,
    d4_multiply,
    transform_kernel,
)
from pcam_repro.models.partial import (
    CyclicLiftingConv2d,
    PartialCyclicGroupConv2d,
    rotate_kernel,
)


@pytest.mark.parametrize(
    ("name", "parameters"),
    [
        ("small_cnn", {"width": 8}),
        ("resnet18", {}),
        ("resnet50", {}),
        ("densenet121", {}),
        ("inception_v3", {}),
        (
            "gdensenet_d4",
            {"initial_channels": 4, "growth_rate": 2, "block_layers": [1, 1]},
        ),
        (
            "pcam_d4loc",
            {
                "initial_channels": 3,
                "growth_rate": 2,
                "stage_layers": [1, 1, 1],
                "transition_channels": [4, 6],
                "fusion_channels": 8,
            },
        ),
        ("partial_se2", {"orientations": 4, "channels": [4, 6]}),
        (
            "vit_small",
            {"embedding_dim": 48, "depth": 2, "heads": 4, "patch_size": 16},
        ),
        (
            "lgvit",
            {"embedding_dim": 32, "depth": 1, "heads": 4},
        ),
        (
            "lotenet",
            {"physical_dim": 3, "bond_dims": [4, 6], "permutation_count": 2},
        ),
        (
            "multiscale_attention",
            {
                "encoder": "small_cnn",
                "encoder_parameters": {"width": 4},
                "scales": [0.75, 1.0],
            },
        ),
    ],
)
def test_classifier_forward_and_backward(name: str, parameters: dict) -> None:
    model = build_model(ModelConfig(name=name, parameters=parameters))
    model.train()
    input_tensor = torch.randn(2, 3, 96, 96)
    logits = model(input_tensor)
    assert logits.shape == (2, 1)
    logits.mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_simclr_wrapper() -> None:
    model = build_model(
        ModelConfig(name="resnet18", parameters={"projection_dim": 32}),
        training_mode="simclr",
    )
    embeddings = model(torch.randn(2, 3, 96, 96))
    assert embeddings.shape == (2, 32)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(2), atol=1e-5)


def test_pretrained_inception_omits_init_weights_override(monkeypatch) -> None:
    constructor_kwargs: dict = {}

    class FakeInception(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 1)
            self.aux_logits = True
            self.AuxLogits = nn.Identity()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.zeros(inputs.shape[0], 4)

    def fake_inception_v3(**kwargs) -> FakeInception:
        constructor_kwargs.update(kwargs)
        return FakeInception()

    monkeypatch.setattr("pcam_repro.models.cnn.models.inception_v3", fake_inception_v3)
    model = InceptionPCam(pretrained=True)

    assert constructor_kwargs["aux_logits"] is True
    assert constructor_kwargs["weights"] is not None
    assert "init_weights" not in constructor_kwargs
    assert model.backbone.aux_logits is False
    assert model.backbone.AuxLogits is None


def test_d4_lifting_rotation_moves_orientation_axis() -> None:
    layer = D4LiftingConv2d(1, 2)
    layer.eval()
    image = torch.randn(1, 1, 25, 25)
    original = layer(image)
    rotated = layer(torch.rot90(image, 1, (-2, -1)))
    permutation = [d4_multiply(d4_inverse(1), g) for g in range(8)]
    expected = torch.rot90(original[:, :, permutation], 1, (-2, -1))
    assert (rotated - expected).abs().mean() < 1e-5


def test_vectorized_d4_group_convolution_matches_reference_and_gradients() -> None:
    layer = D4GroupConv2d(2, 3, bias=True)
    vectorized_input = torch.randn(2, 2, 8, 9, 9, requires_grad=True)
    reference_input = vectorized_input.detach().clone().requires_grad_(True)

    vectorized = layer(vectorized_input)
    reference_groups = []
    for g in range(8):
        inverse_g = d4_inverse(g)
        output_g = sum(
            torch.nn.functional.conv2d(
                reference_input[:, :, h],
                transform_kernel(
                    layer.weight[:, :, d4_multiply(inverse_g, h)],
                    g,
                ),
                padding=layer.padding,
            )
            for h in range(8)
        )
        reference_groups.append(
            output_g + layer.bias[None, :, None, None]
        )
    reference = torch.stack(reference_groups, dim=2)

    assert torch.allclose(vectorized, reference, atol=2e-5, rtol=1e-5)
    gradient = torch.randn_like(vectorized)
    vectorized.backward(gradient)
    vectorized_input_gradient = vectorized_input.grad.detach().clone()
    vectorized_weight_gradient = layer.weight.grad.detach().clone()
    layer.zero_grad(set_to_none=True)
    reference.backward(gradient)
    assert torch.allclose(
        vectorized_input_gradient,
        reference_input.grad,
        atol=2e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        vectorized_weight_gradient,
        layer.weight.grad,
        atol=1e-4,
        rtol=2e-5,
    )


def test_vectorized_partial_lifting_matches_reference_and_gradients() -> None:
    orientations = 16
    layer = CyclicLiftingConv2d(2, 3, orientations)
    batch, spatial_size = 1, 7
    vectorized_input = torch.randn(
        batch,
        2,
        spatial_size,
        spatial_size,
        requires_grad=True,
    )
    reference_input = vectorized_input.detach().clone().requires_grad_(True)
    vectorized = layer(vectorized_input)
    reference_kernel = torch.cat(
        [
            rotate_kernel(
                layer.weight,
                2 * torch.pi * group_index / orientations,
            )
            for group_index in range(orientations)
        ],
        dim=0,
    )
    reference_flat = torch.nn.functional.conv2d(
        reference_input,
        reference_kernel,
        layer.bias.repeat(orientations),
        padding=1,
    )
    reference = reference_flat.reshape(
        batch,
        orientations,
        3,
        spatial_size,
        spatial_size,
    ).permute(0, 2, 1, 3, 4)
    assert torch.allclose(vectorized, reference, atol=2e-5, rtol=1e-5)

    gradient = torch.randn_like(vectorized)
    vectorized.backward(gradient)
    vectorized_input_gradient = vectorized_input.grad.detach().clone()
    vectorized_weight_gradient = layer.weight.grad.detach().clone()
    layer.zero_grad(set_to_none=True)
    reference.backward(gradient)
    assert torch.allclose(
        vectorized_input_gradient,
        reference_input.grad,
        atol=2e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        vectorized_weight_gradient,
        layer.weight.grad,
        atol=1e-4,
        rtol=2e-5,
    )


def test_vectorized_partial_group_convolution_matches_reference_and_gradients() -> None:
    orientations = 16
    layer = PartialCyclicGroupConv2d(2, 3, orientations)
    batch, spatial_size = 1, 5
    vectorized_input = torch.randn(
        batch,
        2,
        orientations,
        spatial_size,
        spatial_size,
        requires_grad=True,
    )
    reference_input = vectorized_input.detach().clone().requires_grad_(True)
    vectorized = layer(vectorized_input)
    gates = layer.gates
    reference_groups = []
    for g in range(orientations):
        angle = 2 * torch.pi * g / orientations
        output_g = sum(
            gates[(h - g) % orientations]
            * torch.nn.functional.conv2d(
                reference_input[:, :, h],
                rotate_kernel(
                    layer.weight[:, :, (h - g) % orientations],
                    float(angle),
                ),
                padding=1,
            )
            for h in range(orientations)
        )
        reference_groups.append(
            output_g + layer.bias[None, :, None, None]
        )
    reference = torch.stack(reference_groups, dim=2)
    assert torch.allclose(vectorized, reference, atol=2e-5, rtol=1e-5)

    gradient = torch.randn_like(vectorized)
    vectorized.backward(gradient)
    vectorized_input_gradient = vectorized_input.grad.detach().clone()
    vectorized_weight_gradient = layer.weight.grad.detach().clone()
    vectorized_gate_gradient = layer.gate_logits.grad.detach().clone()
    layer.zero_grad(set_to_none=True)
    reference.backward(gradient)
    assert torch.allclose(
        vectorized_input_gradient,
        reference_input.grad,
        atol=2e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        vectorized_weight_gradient,
        layer.weight.grad,
        atol=1e-4,
        rtol=2e-5,
    )
    assert torch.allclose(
        vectorized_gate_gradient,
        layer.gate_logits.grad,
        atol=2e-5,
        rtol=1e-5,
    )


def test_partial_convolutions_cache_only_in_evaluation_mode() -> None:
    lifting = CyclicLiftingConv2d(2, 3, orientations=4).eval()
    group = PartialCyclicGroupConv2d(3, 4, orientations=4).eval()
    image = torch.randn(1, 2, 7, 7)
    with torch.inference_mode():
        features = lifting(image)
        first = group(features)
        second = group(features)
    assert lifting._cached_kernel is not None
    assert group._cached_kernel is not None
    assert torch.equal(first, second)

    lifting.train()
    group.train()
    assert lifting._cached_kernel is None
    assert group._cached_kernel is None


def test_partial_model_rebuilds_cached_filters_after_checkpoint_load() -> None:
    reference = build_model(
        ModelConfig(
            name="partial_se2",
            parameters={"orientations": 4, "channels": [4, 6]},
        )
    ).eval()
    restored = build_model(
        ModelConfig(
            name="partial_se2",
            parameters={"orientations": 4, "channels": [4, 6]},
        )
    ).eval()
    image = torch.randn(2, 3, 32, 32)

    with torch.inference_mode():
        restored(image)  # Populate filters for the pre-load parameter state.
        restored.load_state_dict(reference.state_dict())
        restored.eval()  # Mirrors evaluate() after loading best.pt.
        actual = restored(image)
        expected = reference(image)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_d4_classifier_is_rotation_and_reflection_invariant() -> None:
    model = build_model(
        ModelConfig(
            name="gdensenet_d4",
            parameters={"initial_channels": 3, "growth_rate": 2, "block_layers": [1, 1]},
        )
    ).eval()
    image = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        reference = model(image)
        rotated = model(torch.rot90(image, 1, (-2, -1)))
        reflected = model(torch.flip(image, (-1,)))
    assert torch.allclose(reference, rotated, atol=1e-6)
    assert torch.allclose(reference, reflected, atol=1e-6)


def test_pcam_d4loc_exposes_equivariant_evidence_and_invariant_logits() -> None:
    model = build_model(
        ModelConfig(
            name="pcam_d4loc",
            parameters={
                "initial_channels": 3,
                "growth_rate": 2,
                "stage_layers": [1, 1, 1],
                "transition_channels": [4, 6],
                "fusion_channels": 8,
                "dropout": 0.0,
            },
        )
    ).eval()
    image = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        evidence = model.evidence_map(image)
        rotated_evidence = model.evidence_map(torch.rot90(image, 1, (-2, -1)))
        reference = model(image)
        rotated = model(torch.rot90(image, 1, (-2, -1)))
        reflected = model(torch.flip(image, (-1,)))
    assert evidence.shape == (1, 8, 8)
    assert torch.allclose(
        rotated_evidence,
        torch.rot90(evidence, 1, (-2, -1)),
        atol=2e-5,
        rtol=1e-5,
    )
    assert torch.allclose(reference, rotated, atol=2e-5, rtol=1e-5)
    assert torch.allclose(reference, reflected, atol=2e-5, rtol=1e-5)


def test_pcam_d4loc_ablation_controls_are_trainable() -> None:
    model = build_model(
        ModelConfig(
            name="pcam_d4loc",
            parameters={
                "initial_channels": 4,
                "growth_rate": 2,
                "stage_layers": [1, 1, 1],
                "transition_channels": [6, 8],
                "fusion_channels": 8,
                "use_d4": False,
                "center_pooling": False,
                "multiscale": False,
            },
        )
    )
    logits = model(torch.randn(2, 3, 96, 96))
    logits.mean().backward()
    assert logits.shape == (2, 1)
    assert model.evidence_head.weight.grad is not None
