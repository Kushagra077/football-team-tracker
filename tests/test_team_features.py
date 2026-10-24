import numpy as np

from ftt.team.features import colour_feature


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
