# Result files

The numbers in the main README, as written by the scripts in `scripts/`. These are copies of the
files the scripts put in `reports/` (which is not committed: it also holds large videos and caches),
taken after the final runs. Only derived numbers are here, no SoccerNet frames or labels.

| File | Script | Backs this README table |
|---|---|---|
| `tracking_oracle_{val,test}.json` | `run_ablation.py --source oracle` | Trackers: Oracle HOTA, and the oracle numbers quoted in the bullets |
| `tracking_real_{val,test}.json` | `run_ablation.py --source real` | Trackers: Real HOTA / DetA / AssA / IDF1 |
| `det_only_val.json` | `eval_detections.py` | Detector-only DetA at three confidence levels |
| `folds_val.json` | `check_folds.py` | Tuning check on two halves of val (ByteTrack only) |
| `tuning_val.jsonl` | `tune_tracker.py` | The tuning log on val: untuned and tuned HOTA per tracker (the "+1.1..+3.5 HOTA on val" bullet) |
| `team_gt_{val,test}.json` | `eval_team.py --source gt` | Team assignment: ground-truth boxes row |
| `team_real_{val,test}.json` | `eval_team.py --source real` | Team assignment: detector boxes row |
| `team_vote_{val,test}.json` | `eval_team_vote.py` | Team assignment: tracked + vote row, label flips |
| `team_features_{train,val}.json` | `eval_team_features.py` | "Which colour feature?" table |

In `tracking_*.json`, `botsort_gmc_reid__tuned` is BoT-SORT + GMC with appearance ReID (the "Does appearance ReID help?"
table); keys ending in `__tuned` use the shared tuned setting
(`track_high_thresh 0.25`, `match_thresh 0.9`); the keys without it are the untuned defaults.

Not stored as JSON: the goalkeeper counts (right / wrong / unknown) are printed by
`eval_goalkeepers.py`, the coverage counts (111,576 of 111,606 and 434,559 of 434,589) were printed by
`eval_team_vote.py`, and the frame-rate simulation, SportsMOT, hand-labelled and uploaded-clip numbers come
from one-off checks, some on footage that cannot be redistributed.

`team_vote_*.json` holds the accuracy and flips per clip for no vote (`raw`), the default centred vote (`vote`), the
backward-looking vote (`vote_causal`) and a lock reference (`vote_lock15`), plus `mixed_team_tracks`, the number of
tracks that contain boxes from both teams.
