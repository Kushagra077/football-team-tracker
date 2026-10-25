import numpy as np

from ftt.team.features import (
    colour_feature,
    extra_names,
    feature_names,
    histogram_feature,
    lab_pixels,
)


def test_white_is_bright_and_neutral():
    crop = np.full((4, 4, 3), 255, np.uint8)
    lab = colour_feature(crop, np.ones((4, 4), bool))
    assert lab[0] > 99 and abs(lab[1]) < 1 and abs(lab[2]) < 1


def test_blue_has_negative_b():
    crop = np.zeros((4, 4, 3), np.uint8)
    crop[..., 0] = 255  # BGR: pure blue
    assert colour_feature(crop, np.ones((4, 4), bool))[2] < -50


def test_only_kept_pixels_are_averaged():
    crop = np.zeros((2, 2, 3), np.uint8)
    crop[0] = 255  # top row white, bottom row black
    keep = np.array([[True, True], [False, False]])
    assert colour_feature(crop, keep)[0] > 99  # black pixels ignored


def test_empty_mask_gives_nan():
    crop = np.full((2, 2, 3), 255, np.uint8)
    assert np.isnan(colour_feature(crop, np.zeros((2, 2), bool))).all()


def test_box_features_keep_the_input_row_and_drop_all_grass_crops():
    import cv2
    import pandas as pd

    from ftt.team.features import box_features

    cfg = {
        "crop": {"y_top": 0.15, "y_bottom": 0.50, "x_trim": 0.20},
        "grass_mask": {"h": [34, 42], "s_min": 40, "v_min": 40},
    }
    img = np.zeros((200, 200, 3), np.uint8)
    img[:, :100] = (255, 255, 255)  # left half white: a "shirt"
    img[:, 100:] = cv2.cvtColor(np.uint8([[[38, 150, 150]]]), cv2.COLOR_HSV2BGR)[0, 0]  # pitch
    boxes = pd.DataFrame(
        {
            "frame": [1, 1],
            "x1": [10.0, 110.0],
            "y1": [20.0, 20.0],
            "x2": [90.0, 190.0],
            "y2": [120.0, 120.0],
        },
        index=[7, 9],
    )
    out = box_features(boxes, lambda f: img, cfg)
    assert out.index.tolist() == [7]  # row 9 is all pitch: dropped; row 7 keeps its own index
    assert out.loc[7, "L"] > 99


def _pixels(*colours_bgr):
    """Lab pixels of a strip made of equal-sized bands of the given BGR colours."""
    strip = np.zeros((4, 4 * len(colours_bgr), 3), np.uint8)
    for i, c in enumerate(colours_bgr):
        strip[:, 4 * i : 4 * i + 4] = c
    return lab_pixels(strip, np.ones(strip.shape[:2], bool))


def test_histogram_is_unit_length_and_one_bin_for_one_colour():
    h = histogram_feature(_pixels((0, 0, 220)))
    assert len(h) == len(feature_names({"features": {"kind": "hist_lab"}}))
    assert np.isclose((h**2).sum(), 1.0) and (h > 0).sum() == 1


def test_striped_shirt_keeps_two_peaks_where_the_mean_is_neither_colour():
    red, white = (0, 0, 220), (255, 255, 255)
    stripes = _pixels(red, white)
    h = histogram_feature(stripes)
    assert (h > 0).sum() == 2 and np.allclose(h[h > 0], np.sqrt(0.5))
    mean = stripes.mean(axis=0)
    for plain in (_pixels(red), _pixels(white)):  # the mean sits between the two stripes
        assert np.linalg.norm(mean - plain.mean(axis=0)) > 20


def test_histogram_of_no_pixels_is_nan():
    assert np.isnan(histogram_feature(np.empty((0, 3)))).all()


def test_box_features_returns_the_configured_columns():
    import pandas as pd

    from ftt.team.features import box_features

    cfg = {
        "crop": {"y_top": 0.15, "y_bottom": 0.50, "x_trim": 0.20},
        "grass_mask": {"h": [34, 42], "s_min": 40, "v_min": 40},
        "features": {"kind": "hist_lab"},
    }
    img = np.full((200, 200, 3), 255, np.uint8)
    boxes = pd.DataFrame({"frame": [1], "x1": [10.0], "y1": [20.0], "x2": [90.0], "y2": [120.0]})
    out = box_features(boxes, lambda f: img, cfg)
    # the mean Lab always comes along (the colour-outlier rule uses it), then the histogram
    assert list(out.columns) == ["frame", "x1", "y1", "x2", "y2", *extra_names(cfg)]
    assert extra_names(cfg)[:3] == ["L", "a", "b"] and out.loc[0, "L"] > 99
    assert np.isclose((out[feature_names(cfg)].to_numpy() ** 2).sum(), 1.0)
