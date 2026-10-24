import cv2
import numpy as np
import pytest

from ftt.team.crops import grass_mask, torso_crop

CFG = {
    "crop": {"y_top": 0.15, "y_bottom": 0.50, "x_trim": 0.20},
    "grass_mask": {"h": [34, 42], "s_min": 40, "v_min": 40},
}


def _bgr_from_hsv(h, s, v):
    return cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]


def test_torso_crop_takes_the_shirt_band():
    frame = np.zeros((200, 200, 3), np.uint8)
    # Box 100 wide x 100 tall at (50, 50): band = rows 65..100, cols 70..130.
    crop = torso_crop(frame, np.array([50.0, 50.0, 150.0, 150.0]), CFG)
    assert crop.shape == (35, 60, 3)


def test_torso_crop_is_the_right_pixels():
    frame = np.zeros((200, 200, 3), np.uint8)
    frame[65:100, 70:130] = 255
    crop = torso_crop(frame, np.array([50.0, 50.0, 150.0, 150.0]), CFG)
    assert (crop == 255).all()


def test_torso_crop_clamps_at_image_edge():
    frame = np.zeros((100, 100, 3), np.uint8)
    # Box hangs off the left and top: must not wrap around or raise.
    crop = torso_crop(frame, np.array([-40.0, -20.0, 60.0, 80.0]), CFG)
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert crop.shape[0] <= 100 and crop.shape[1] <= 100


def test_grass_mask_drops_pitch_green():
    pitch = np.full((4, 4, 3), _bgr_from_hsv(38, 150, 150), np.uint8)
    assert not grass_mask(pitch, CFG).any()


def test_grass_mask_keeps_green_shirt_outside_pitch_hue():
    # Green shirt measured around hue 46-50 on SNMOT-064; must survive the narrow range.
    shirt = np.full((4, 4, 3), _bgr_from_hsv(50, 150, 150), np.uint8)
    assert grass_mask(shirt, CFG).all()


@pytest.mark.parametrize("hsv", [(113, 200, 200), (0, 0, 255), (20, 0, 20)])  # blue, white, dark
def test_grass_mask_keeps_non_green(hsv):
    px = np.full((4, 4, 3), _bgr_from_hsv(*hsv), np.uint8)
    assert grass_mask(px, CFG).all()


def test_grass_mask_keeps_low_saturation_at_pitch_hue():
    # Grey-ish pixel with a green-range hue is not grass: s/v floors protect it.
    px = np.full((4, 4, 3), _bgr_from_hsv(38, 10, 200), np.uint8)
    assert grass_mask(px, CFG).all()
