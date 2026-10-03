# data/

Nothing large lives here. SoccerNet-Tracking frames are on Kaggle; this folder
only holds tiny clips for local dev and tests (`samples/`, gitignored contents).

Layout the code expects (override with `configs/paths.yaml`):

    data/samples/<clip_id>/img1/000001.jpg ...
    data/samples/<clip_id>/gt/gt.txt
    data/samples/<clip_id>/gameinfo.ini

Cached detector output goes in `../cache/detections/`, not here.
Attribution: SoccerNet (CC BY-NC 4.0); cite Cioppa et al., CVPR 2022 (SoccerNet-Tracking).
