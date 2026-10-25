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


def test_device_default_is_looked_up_per_call_not_at_import():
    # On ZeroGPU, CUDA only exists inside the GPU call: a default frozen at import would be "cpu".
    import inspect

    from ftt.detect.run_detector import run_detector
    from ftt.pipeline import process_video
    from ftt.video import track_video

    for fn in (process_video, track_video, run_detector):
        assert inspect.signature(fn).parameters["device"].default is None
