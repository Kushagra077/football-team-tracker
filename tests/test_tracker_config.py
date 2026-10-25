from pathlib import Path

import pytest

from ftt.track.config import load_tracker_cfg, load_tuned

CONFIGS = Path(__file__).parents[1] / "configs"


def test_tuned_setting_is_the_one_in_the_readme():
    assert load_tuned(CONFIGS) == {"track_high_thresh": 0.25, "match_thresh": 0.9}


@pytest.mark.parametrize("name", ["botsort_gmc", "bytetrack", "ocsort"])
def test_tuned_values_override_the_trackers_own_config_and_the_rest_is_kept(name):
    own = load_tracker_cfg(CONFIGS, name, tuned=False)
    cfg = load_tracker_cfg(CONFIGS, name)
    assert (cfg["track_high_thresh"], cfg["match_thresh"]) == (0.25, 0.9)
    assert (own["track_high_thresh"], own["match_thresh"]) == (0.5, 0.8)
    assert {k: v for k, v in cfg.items() if k not in load_tuned(CONFIGS)} == {
        k: v for k, v in own.items() if k not in load_tuned(CONFIGS)
    }
