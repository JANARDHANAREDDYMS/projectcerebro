from __future__ import annotations

import torch

from ml_core.models.eegnet import EEGNet
from ml_core.models.shallowconv import ShallowConvNet


def test_shallowconv_forward_shape():
    model = ShallowConvNet()
    logits = model(torch.randn(4, 1, 5, 512))
    assert tuple(logits.shape) == (4, 3)


def test_eegnet_forward_and_embedding_shape():
    model = EEGNet()
    x = torch.randn(4, 1, 5, 512)
    logits = model(x)
    embedding = model.encode(x)
    assert tuple(logits.shape) == (4, 3)
    assert tuple(embedding.shape) == (4, 128)
