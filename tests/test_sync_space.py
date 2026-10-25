import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "sync_space", Path(__file__).parents[1] / "scripts" / "sync_space.py"
)
sync_space = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync_space)


def test_sync_copies_the_package_and_configs_and_drops_stale_files(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "ftt").mkdir()
    (tmp_path / "ftt" / "stale.py").write_text("old")
    sync_space.sync(tmp_path)
    assert (tmp_path / "ftt" / "pipeline.py").exists()
    assert not (tmp_path / "ftt" / "stale.py").exists()
    assert (tmp_path / "configs" / "team.yaml").exists()
    assert (tmp_path / "configs" / "botsort_gmc.yaml").exists()
    assert not list((tmp_path / "ftt").rglob("__pycache__"))


def test_sync_refuses_a_folder_that_is_not_a_git_clone(tmp_path):
    with pytest.raises(SystemExit):
        sync_space.sync(tmp_path)
