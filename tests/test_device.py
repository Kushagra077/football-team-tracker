import torch

from ftt.device import default_device


def test_default_device_is_one_torch_accepts():
    assert default_device() in {"cuda", "mps", "cpu"}
    torch.zeros(1, device=default_device())


def test_cuda_wins_then_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert default_device() == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert default_device() == "cpu"
