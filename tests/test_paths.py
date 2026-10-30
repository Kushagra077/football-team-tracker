from pathlib import Path

from ftt.paths import DETECTOR_ENV, load_paths

ROOT = Path(__file__).parents[1]


def test_paths_default_is_the_pinned_detector(monkeypatch):
    monkeypatch.delenv(DETECTOR_ENV, raising=False)
    assert load_paths(ROOT)["detector_weights"].endswith("football_detection_v6.pt")


def test_env_var_switches_the_detector_in_the_same_folder(monkeypatch):
    monkeypatch.setenv(DETECTOR_ENV, "football_detection_v3")
    pinned = Path(load_paths(ROOT)["detector_weights"])
    assert pinned.name == "football_detection_v3.pt"
    assert pinned.parent.name == "models"  # same folder as the pinned one


def test_env_var_accepts_a_name_with_extension(monkeypatch):
    monkeypatch.setenv(DETECTOR_ENV, "football_detection_v7.pt")
    assert Path(load_paths(ROOT)["detector_weights"]).name == "football_detection_v7.pt"
