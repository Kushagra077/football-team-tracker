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
