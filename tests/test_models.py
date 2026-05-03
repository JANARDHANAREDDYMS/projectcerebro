"""Forward-pass shape tests for ShallowConvNet and EEGNet."""
from __future__ import annotations

import torch

from ml_core.models import EEGNet, ShallowConvNet


def test_shallowconv_forward_shape():
    model = ShallowConvNet(n_classes=3)
    x = torch.randn(4, 1, 5, 512)
    out = model(x)
    assert out.shape == (4, 3)


def test_eegnet_forward_shape():
    model = EEGNet(n_classes=3, embed_dim=128)
    x = torch.randn(4, 1, 5, 512)
    out = model(x)
    assert out.shape == (4, 3)


def test_eegnet_embedding_shape():
    model = EEGNet(n_classes=3, embed_dim=128)
    x = torch.randn(4, 1, 5, 512)
    logits, emb = model(x, return_embedding=True)
    assert logits.shape == (4, 3)
    assert emb.shape == (4, 128)


def test_eegnet_replace_classifier():
    model = EEGNet(n_classes=3)
    model.replace_classifier(n_classes=4)
    x = torch.randn(2, 1, 5, 512)
    out = model(x)
    assert out.shape == (2, 4)


def test_shallowconv_param_count_sane():
    model = ShallowConvNet()
    n_params = sum(p.numel() for p in model.parameters())
    assert 10_000 < n_params < 5_000_000  # sanity bound


def test_eegnet_param_count_sane():
    model = EEGNet()
    n_params = sum(p.numel() for p in model.parameters())
    assert 1_000 < n_params < 5_000_000
