# data/

Nothing large lives here, and no SoccerNet data is committed. SoccerNet-Tracking is obtained from
[soccer-net.org](https://www.soccer-net.org/) under its NDA (non-commercial research, no redistribution).

Where the code looks for it (paths are set in `configs/paths.yaml`):

    data/raw_meta/{train,test}/<clip>/gt/gt.txt, gameinfo.ini, seqinfo.ini   # SoccerNet text files
    ../football-detect-serve/dataset/images/{train,val,test}/<clip>_<frame:06d>.jpg   # frames (A1's layout)

`val` is not a SoccerNet folder: it is every 5th sorted train clip (the same rule A1 used), so the detector
never trained on it. `data/samples/` is for tiny local test clips and is gitignored.

Cached detector output goes in `cache/` (gitignored), not here.
Attribution: cite Cioppa et al., "SoccerNet-Tracking", CVPR Workshops 2022.
