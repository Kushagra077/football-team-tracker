<p align="center">
  <img src="reports/figures/banner.svg" alt="Football Team Tracker: player tracking and team assignment from broadcast video" width="100%">
</p>

A computer-vision pipeline that follows every player in a football broadcast with a fixed ID and splits them into
the two teams by shirt colour, with seven trackers benchmarked on SoccerNet-Tracking.

<p align="center">
  <a href="https://youtu.be/YrP9KGjes4g">
    <img src="reports/figures/demo.gif" alt="Five seconds of the demo: players tracked and coloured by team" width="720">
  </a>
  <br>
  <sub><b>▶ Click for the full 2-minute demo on YouTube</b> (unlisted). Five seconds of a broadcast highlights clip run
  through <code>scripts/process_video.py</code>: orange and cyan are the two teams, white is "no team" (keepers, referees).
  Footage © its rights holder, shown as a non-commercial portfolio demo.<br>
  The numbers drawn on players are <b>tracker IDs, not jersey numbers</b>: the tracker numbers each new track it starts
  (so #122 is simply the 122nd track), and reading shirt numbers is a separate, later task.</sub>
</p>

<p align="center"><b>Try it live:</b> <a href="https://huggingface.co/spaces/Kushagra77/football-detect-serve">Hugging Face Space</a> (upload a short football clip on the <i>Track (video)</i> tab; runs on a free ZeroGPU, so a visitor gets a few minutes of GPU per day, and only the first 20 s of a clip are used).</p>

| Headline result (SoccerNet-Tracking test split, 49 clips, best tracker BoT-SORT + GMC) | |
|---|---|
| Tracking score (HOTA) with a perfect detector (**oracle**) | **84.2** |
| Tracking score (HOTA) with my own detector (**real**) | **59.0** |
| Players given the correct team (tracked + voted, per detection; 98.5% without the vote, see below) | **98.2%** |
| Speed: detector + tracker + camera-motion compensation + colour features, 1080p, no rendering, MacBook Pro (Apple M2, 8 GB, PyTorch MPS) | **~35-45 fps** |

## What this project does

Give it a football broadcast video. It finds every player, follows each one with a fixed ID number as they move,
and splits them into the two teams by shirt colour. The output is an annotated video and a table (one row per player
per frame) that later projects can use for tactics.

It also answers a measurement question: **how much tracking quality is lost because the detector is imperfect?**
Every tracker is run twice, once on the true player boxes (oracle) and once on my detector's boxes (real), and the
difference is the detector's cost.

<p align="center">
  <img src="reports/figures/tracking_trails.svg" alt="Two seconds of tracked players drawn as trails coloured by team" width="90%">
  <br>
  <sub>Two seconds of the demo, drawn from the pipeline's output table only (feet positions, no video pixels):
  orange and cyan are the two teams, white is "no team", numbers are tracker IDs (not jersey numbers).</sub>
</p>

## How to read the numbers

| Term | Meaning |
|---|---|
| Oracle | The tracker gets the true player boxes (a perfect detector): the best it could possibly do. |
| Real | The tracker gets my own detector's boxes: what you get in practice. |
| HOTA (0-100) | The main tracking score: finds the players **and** keeps the same ID on each one. |
| DetA | Detection part of HOTA: were the players found? |
| AssA | Association part of HOTA: did each player keep his ID over time? |
| IDF1 | Identity F1 score over the whole sequence: balances identity precision and recall, so it measures how well detections carry the correct ground-truth identity. Higher is better. |
| IDSW | ID switches: times a player's number jumped to someone else. Lower is better. |
| Detector cost | Oracle HOTA minus Real HOTA: how much the imperfect detector loses. |

## How it works

```mermaid
flowchart LR
    V[video frames] --> D[my YOLO detector<br/>conf 0.1]
    D --> T[tracker<br/>BoT-SORT + GMC]
    T --> C[torso crop<br/>minus grass pixels]
    C --> F[Lab colour histogram]
    F --> K[KMeans k=2<br/>fit on first 50 pitch frames]
    K --> R[rolling vote<br/>per track, 25 frames]
    R --> G[goalkeeper rule]
    G --> P[tracks.parquet<br/>+ annotated mp4]
```

1. **Detect.** My YOLO detector ([football-detect-serve](https://github.com/Kushagra077/football-detect-serve), v3, yolo26n) finds players, goalkeepers and referees. Confidence is kept low (0.1) so the
   tracker's second pass can use weak boxes.
2. **Track.** Ultralytics' tracker classes give each box an id. Evaluation feeds cached boxes to the tracker; uploaded
   video goes through `model.track(stream=True)`, and both give identical output on one val clip's frames.
3. **Colour.** For each player box: take the torso band (15-50% of the height, 20% trimmed off each side), drop grass
   pixels (HSV hue 34-42), and describe what is left as a colour histogram in Lab space (4 x 6 x 6 bins; distances in Lab
   match how different colours look). A histogram keeps both stripes of a striped shirt, where a mean would blend them into
   a colour neither stripe has. The mean Lab colour is still computed, for the referee rule in step 4.
4. **Cluster.** KMeans with k=2 on the players of the first 50 frames that show at least 3 players. No labels needed.
   On uploaded video, only wide pitch-view frames are used and labelled (close-ups get no team), and a box whose mean
   colour fits neither team (a black-kit referee) gets no team.
5. **Vote.** Each track's label is the majority of its last 25 frames. There is no permanent lock: trackers sometimes move
   an id onto another player, and a lock would keep the wrong team for the rest of the clip (lock cost 6-8 accuracy points).
6. **Goalkeepers.** A keeper's colour matches neither team, so a keeper joins the team whose players stand nearer in the
   image (median x). If the teams overlap or the keeper is in between, the answer is "unknown" (null), not a guess.
7. **Export.** `tracks.parquet` (see [Output format](#output-format)) plus an annotated mp4.

## Results

### Trackers (SoccerNet-Tracking test split, 49 clips, run once)

Settings were chosen on the 11 validation clips only; the test split was run once at the end with them fixed.

**How it was scored.** Metrics come from [TrackEval](https://github.com/JonathonLuiten/TrackEval)'s HOTA, CLEAR and
Identity classes, called through my own thin wrapper (`src/ftt/eval/trackeval_wrap.py`), so the numbers are not
directly comparable to the SoccerNet leaderboard. Players, goalkeepers and referees are scored together as persons
(class labels are ignored; the ball and "other" are excluded). **No leakage:** the detector was never trained on
any validation or test clip (checked by `scripts/prepare_data.py`). The detector's training did use the validation clips to pick a
checkpoint, so validation numbers may be slightly optimistic; the numbers below are from the test split.

<p align="center">
  <img src="reports/figures/tracker_benchmark.svg" alt="Oracle vs real HOTA for seven trackers" width="90%">
</p>

| Tracker | Oracle HOTA | Real HOTA | Real DetA | Real AssA | Real IDF1 | Detector cost |
|---|---|---|---|---|---|---|
| **BoT-SORT + GMC** | **84.2** | **59.0** | **70.9** | **49.4** | **68.5** | 25.2 |
| BoT-SORT (no GMC) | 80.4 | 56.6 | 69.9 | 46.0 | 65.2 | 23.9 |
| Deep OC-SORT + GMC | 76.6 | 56.0 | 69.0 | 45.6 | 65.4 | 20.6 |
| OC-SORT (+ BYTE low-conf pass) | 75.2 | 55.4 | 68.0 | 45.3 | 65.5 | 19.8 |
| ByteTrack | 74.1 | 55.3 | 68.1 | 45.1 | 65.2 | 18.8 |
| OC-SORT | 75.2 | 55.2 | 68.1 | 44.9 | 65.2 | 20.1 |
| FastTracker | 74.1 | 55.0 | 68.2 | 44.5 | 64.4 | 19.2 |

- **The detector costs about 19-25 HOTA points**, e.g. BoT-SORT+GMC 84.2 -> 59.0 and ByteTrack 74.1 -> 55.3.
  The split is approximate, not additive: a missed box hurts differently depending on the tracker's settings.
- **Detection and association both drop by about 18-25 points** (e.g. BoT-SORT+GMC DetA 95.4 -> 70.9, AssA 74.4 -> 49.4).
  Association ends up the weaker of the two (45-49 vs 68-71).
- **Camera-motion compensation (GMC) matters for broadcast video.** It is BoT-SORT+GMC's edge: on oracle boxes it cuts
  ID switches from 3524 to 1251.
- **The bottom four trackers (55.0-55.4 real HOTA) are within 0.4 HOTA** and swap places between val and test, so treat them as equal.
- Tuning (one shared setting for all trackers, stored once in `configs/tuned.yaml`: `track_high_thresh 0.25`, `match_thresh 0.9`) gained +1.1..+3.5 HOTA on
  val and +0.3..+2.7 on test (val log: `results/tuning_val.jsonl`).
- **Tuning has a cost: more ID switches.** On test the shared setting raises ID switches for 6 of 7 trackers (e.g.
  BoT-SORT+GMC 2398 -> 2705, OC-SORT 3525 -> 3805; FastTracker is the exception, 3368 -> 3271). It lowers the first-pass
  confidence from 0.5 to 0.25, so weaker boxes are tracked: DetA and HOTA go up, switches too.
- **The detector is the bottleneck, not the association step.** On val the detector alone reaches DetA 68.3 / 72.8 / 73.1
  at confidence 0.1 / 0.25 / 0.5 (no tracker), and BoT-SORT+GMC's tracked DetA is 74.2 (untuned): the tracker finds about
  as many players as the detector does.

### Team assignment

| What is scored | Val (11 clips) | Test (49 clips) |
|---|---|---|
| Ground-truth boxes, per detection / per player (A) | 97.5% / 100% | 97.6% / 99.9% |
| Detector boxes (matched to GT players), per detection / per player (A) | 99.0% / 100% | 98.7% / 99.9% |
| Tracked, colour per frame (no vote), per detection (B) | 98.7% | 98.5% |
| Tracked + vote, per detection (B) | 98.6% | 98.2% |
| Coverage of the two tracked rows: tracked players matched to a GT player that got a label | 99.97% (111,576 of 111,606) | 99.99% (434,559 of 434,589) |
| Label flips: raw -> vote | 733 -> 147 | 3693 -> 852 (-77%) |
| Goalkeepers: right / wrong / unknown (C) | 8 / 1 / 7 of 16 | 21 / 5 / 37 of 63 |

How each group was measured: **(A)** detector confidence 0.5, every 25th frame, boxes matched to ground truth at IoU 0.5.
**(B)** the pipeline's own settings: detector confidence 0.1, BoT-SORT+GMC, every frame. **(C)** team positions are taken
from the ground-truth labels, so this scores the goalkeeper rule on its own, not the whole pipeline.

**Does the vote help?** It trades a little accuracy for much less flicker: **-0.1 points on val and -0.3 on test, for 77%
fewer label flips**. On test the vote is worse on 34 of 49 clips and better on 15. A likely reason: 21% of tracks (491 of
2,344 on test) contain boxes from both teams (the tracker's id jumped between players), and a vote that only looks back keeps
the old label for a while after each jump. I measured the 21%, not the cause. A window centred on each frame (possible offline) might fix this; I have not
tried it.

The detector rows only score players the detector found, so they are not "better than GT". Cluster ids are arbitrary,
so accuracy uses the better of the two cluster-to-side mappings.

**Which colour feature?** The first version described a shirt by its mean Lab colour. I compared it with five alternatives
on ground-truth boxes (same crop, same KMeans, same scoring; only the feature changes), tuning on the 46 train clips that
are neither val nor test, then confirming on val:

| Feature | Train (46 clips) | Val (11 clips) |
|---|---|---|
| mean Lab (first version) | 96.8% | 96.9% |
| mean of a and b only (drop brightness, to ignore shadows) | 92.4% | 93.9% |
| median Lab | 95.3% | 95.1% |
| mean + standard deviation of Lab | 96.9% | 97.0% |
| **Lab colour histogram, 4 x 6 x 6 bins (used now)** | **97.5%** (better on 21 clips, worse on 3, by 0.5 points or more) | **97.5%** (better on 4, worse on 0, by 0.5 points or more) |
| histogram of a and b only | 96.3% | 95.6% |

What the table shows:

- Dropping brightness is clearly worse (a white and a dark kit then look alike).
- The histogram's bin size hardly matters (3x5x5 to 5x8x8 gave 97.3-97.5% on train).
- On the full pipeline (real boxes, tracker, vote) the histogram also wins: val 98.2% -> 98.6%, test 97.8% -> 98.2%.
- The feature was chosen on train and val only. The test column is the final choice, and 97.8% was the earlier
  mean-colour result.
- One team setting did change after a test result: the rule for which frames the two colours are learned from (one test
  clip opens on a non-pitch shot and crashed the first run). It changes no val clip, and tracker scores are untouched.

SoccerNet has few striped kits, so I also hand-labelled boxes from three amateur clips with striped and similar-blue kits:

- 98 boxes in total, labelled kit A or B (keepers, referees and unclear boxes skipped), scored with the same clustering.
- Mean Lab 68% (58 / 80 / 66% per clip) vs histogram 79% (77 / 74 / 84%).
- One labeller and about 30 boxes per clip, so treat it as a direction (the histogram helps on two clips and loses on
  one), not a benchmark.

**It is not better everywhere.** On the Chelsea demo (a floodlit broadcast with close-ups):

- I hand-labelled 56 player boxes: mean colour 100%, histogram 98.2% (one box).
- Where the two versions disagree (3.2% of the clip), the mean colour was right on 23 and the histogram on 13 of the 36
  boxes I could judge by eye.
- The histogram turns some dark-blue shirts in shade or motion blur orange, including one player in the 38 s close-up.
  Different brightness bins (2, 3 or 6 instead of 4) did not change this.
- The YouTube demo was rendered before the change, with the mean colour, and I kept it because it colours that close-up
  correctly.
- The histogram is also worse for the referee rule: its distances separate a black-kit referee from players less well than
  mean-colour distances do. That rule therefore still uses the mean Lab colour on its own two-colour clustering.

**How to read accuracy with coverage.** The accuracy is measured on boxes that received a label.

- **SoccerNet:** nearly all boxes get one (99.97-99.99%; the rest had an empty or all-grass torso crop), because the
  abstention rules below belong to the upload path and are not applied in these numbers.
- **Uploaded video:** the system abstains on purpose, so coverage is lower. Of the player boxes it tracked, **92.7% got a
  team on the Chelsea demo, 97.0% on the 22 s SportsMOT clip and 94.1% on the 31 s clip**; the rest were close-ups, colours
  that fit neither team, or unusable crops.
- **Goalkeepers** are separate: on the demo none of the 702 keeper boxes got a team (the rule abstained), and on the 31 s
  clip one keeper got one, wrongly.
- Accuracy on the uploaded clips was not measured (no team labels), so the coverage figures say how often a team is shown,
  not how often it is right.

### Out-of-domain check: SportsMOT

The same one-command pipeline (`process_video.py`, settings unchanged) on 2 football clips from
[SportsMOT](https://github.com/MCG-NJU/SportsMOT)'s val split: a different broadcaster and stadium, 720p instead of 1080p,
53 s in total. Scored against SportsMOT's player boxes (players and goalkeepers kept).

| Clip | HOTA | DetA | AssA | IDF1 | IDSW |
|---|---|---|---|---|---|
| v_ITo3sCnpw_k_c010 (22 s) | 67.3 | 74.0 | 61.4 | 78.8 | 28 |
| v_ITo3sCnpw_k_c011 (31 s, fast pan + blur) | 46.9 | 55.9 | 40.2 | 51.1 | 145 |
| **Both** | **56.1** | **63.1** | **50.3** | **63.1** | 173 |

- **Tracking transfers about as well as on SoccerNet** (BoT-SORT+GMC real HOTA 59.0 on SoccerNet test), with only 2 clips,
  so this is a sanity check, not a benchmark. The two datasets also draw boxes slightly differently.
- **Team colours:** SportsMOT has no team labels, so this is checked by eye only (with the first, mean-colour version).
  - The histogram then cut label flips from 13 to 7 and from 50 to 28 on the two clips (not re-inspected by eye).
  - Most players are coloured correctly; a few red shirts flip on the 22 s clip.
  - On the 31 s clip the goalkeeper rule decides one keeper and gets him wrong (the camera sits at his goal, where the other
    team's players are, so image position misleads).
  - The referee in black is detected as a "player". The colour rule below now gives his boxes no team about a third of the
    time (22-34% of non-player boxes), at a cost of 0.2-1.0% of real players.
- No SportsMOT frames are published here (its terms forbid redistribution without permission).

## Setup

Python 3.13 and [uv](https://docs.astral.sh/uv/). The detector weights come from my detector project, football-detect-serve, which is
public and must be cloned next to this repo (`configs/paths.yaml` points there):

```bash
git clone https://github.com/Kushagra077/football-team-tracker
git clone https://github.com/Kushagra077/football-detect-serve   # detector: provides models/football_detection_v3.pt
cd football-team-tracker
uv sync --extra dev
uv run ruff check .
uv run pytest -m "not integration"            # no data needed; data-dependent tests skip
```

## Run it on a video

```bash
uv run python scripts/process_video.py path/to/clip.mp4            # add --device cpu without an Apple GPU
#   -> reports/<detector>/process_video/<clip>/tracks.parquet + annotated.mp4
```

Wide broadcast views work best. Ids restart after camera cuts, and close-ups get no team (see limitations).

### Or run it in Docker (no Python setup)

```bash
docker build -t football-team-tracker .
docker run --rm football-team-tracker                      # runs the tests
docker run --rm -v "$PWD/clips:/clips" -v "$PWD/out:/app/reports" football-team-tracker \
    python scripts/process_video.py /clips/my_clip.mp4 --device cpu
#   -> out/football_detection_v3/process_video/my_clip/tracks.parquet + annotated.mp4
```

- The image has the locked dependencies and the detector weights (downloaded from my detector repo at a pinned commit).
- It is **CPU only** (about 12 fps on my MacBook, 150 frames of 720p in 12 s): the lockfile's GPU libraries are left out
  and the CPU build of the same PyTorch version is installed instead.
- The 16 tests that need SoccerNet files or local weights skip inside the container, the same as in CI.

### Or try it in the browser

The [Hugging Face Space](https://huggingface.co/spaces/Kushagra77/football-detect-serve) has a **Track (video)** tab
that runs the same `process_video` function on a free ZeroGPU (`scripts/sync_space.py` copies `src/ftt` into the Space).
Limits of the free demo:

- Only the first 20 s of a clip are used, and wider uploads are shrunk to 1280 px.
- Each visitor has a small daily GPU quota. A 20 s, 25 fps clip is tracked in about 12-20 s on the GPU (12 s before the
  colour histogram, 19 s right after it; the speed fix since then is not timed on the Space).
- In my tests each run took roughly half a minute off the quota meter (probably including GPU start-up), so expect around
  ten runs a day when signed in.
- The Space runs Python 3.12 and PyTorch 2.8-2.13; this repo is developed on Python 3.13 and PyTorch 2.14. The two setups
  were not compared output by output.
- The Space's other tab, **Detect (image)**, is the detector on its own.

Timing on a MacBook Pro (Apple M2, 8 GB, PyTorch MPS, Ultralytics 8.4.183, image size 640, one frame at a time): the
2-minute 1080p demo (3000 frames) took 75 s for detection, tracking, camera-motion compensation and colour features
(40 fps) and 27 s to draw the annotated video, 1 min 50 s in total. A 31 s 720p clip ran at 43 fps in the same stage.

## Output format

`tracks.parquet`, one row per tracked box per frame, validated against the schema in `src/ftt/schema.py` on write:

| Column | Type | Meaning |
|---|---|---|
| `frame_id` | int | Frame number, starting at 1 |
| `timestamp` | float | Seconds (`frame_id / fps`) |
| `track_id` | int | Tracker ID (not the jersey number), stable while the player stays in shot |
| `team` | Int8, nullable | 0 or 1 (cluster ids, arbitrary per clip); null for referees, undecided keepers, close-ups |
| `role` | category | `player`, `goalkeeper` or `referee` (from the detector) |
| `x1`, `y1`, `x2`, `y2` | float | Box in image pixels |
| `conf` | float | Detector confidence |
| `jersey_number`, `x_pitch`, `y_pitch` | nullable | Always null here (null, never 0); later projects fill them |
| `is_ball` | bool | Always false (the ball is a separate project) |

## Reproducing the benchmarks

SoccerNet-Tracking is available from [soccer-net.org](https://www.soccer-net.org/) after signing its NDA; it is not
redistributed here. `configs/paths.yaml` expects the original text files under `data/raw_meta/{train,test}/<clip>/`
and the frames in the detector project's layout (`../football-detect-serve/dataset/images/{train,val,test}/<clip>_<frame>.jpg`).

```bash
uv run python scripts/cache_detections.py --split test                  # detector boxes, cached once
uv run python scripts/run_ablation.py --source oracle --split test \
    --configs botsort_gmc botsort_nogmc deepocsort_gmc ocsort_byte bytetrack ocsort fasttrack \
    --set track_high_thresh=0.25 match_thresh=0.9 --tag tuned
uv run python scripts/run_ablation.py --source real --split test \
    --configs botsort_gmc botsort_nogmc deepocsort_gmc ocsort_byte bytetrack ocsort fasttrack \
    --set track_high_thresh=0.25 match_thresh=0.9 --tag tuned
uv run python scripts/eval_team_vote.py --split test
uv run python scripts/make_figures.py                                   # README charts from the result files
```

Tuning happens on val only (`scripts/tune_tracker.py` refuses other splits).

The result files behind the README tables are committed in [`results/`](results/) (see its README for which file backs which
table), so the numbers can be checked without the SoccerNet data.

Layout: `src/ftt/` (data, detect, track, team, eval, schema, export, render, video, scale, pipeline), `scripts/` (one command per
experiment), `configs/` (tracker YAMLs, `team.yaml`), `tests/` (CI runs ruff and pytest on every push).

## Design decisions

The choices that shaped the project, what each one bought, and what it cost.

| Decision | Why | Cost / trade-off |
|---|---|---|
| **Score every tracker twice: true boxes (oracle) and my detector's boxes (real)** | Separates "the tracker is weak" from "the detector is weak". The gap (19-25 HOTA points) is the detector's cost, so the next improvement is obvious. | Every experiment runs twice. |
| **Tune on val only, run the test split once** (`tune_tracker.py` refuses other splits); **one shared setting for all trackers** | Test numbers are not tuned on, and no tracker gets a private advantage. | A per-tracker setting might lift some of them a little. |
| **Use Ultralytics' trackers instead of writing my own** | The question is how they compare on football with my detector, not who can re-implement ByteTrack. | The code is AGPL-3.0, so this repo is too. |
| **BoT-SORT with camera-motion compensation (GMC)** | Broadcast cameras pan and zoom. On true boxes GMC cut ID switches from 3524 to 1251. | It still loses IDs at camera cuts: there is no re-identification across shots. |
| **Team colour from the torso minus grass, KMeans with k=2, no labels** | No team labels are needed, so it works on any footage. A per-track rolling vote cut label flips by 77% on the test split. | Only two colours, and teams in similar kits are not handled. The vote costs 0.3 accuracy points on test (see "Does the vote help?"). |
| **Lab colour histogram instead of the mean colour** | A mean blends a striped shirt into a colour neither stripe has. Chosen on train clips, confirmed on val: better on SoccerNet and on two of my three amateur clips. | Worse on dark-blue shirts in shade or blur (Chelsea demo). It is one line in `configs/team.yaml` to switch back to `mean_lab`. |
| **Abstain rather than guess** (pitch-view check, colour-outlier rule, goalkeeper rule) | A wrong team colour on screen looks worse than no colour. | Lower coverage on uploads (92.7-97.0% of boxes got a team), and the keeper rule abstains on more than half the test keepers. |
| **One `process_video` function** for the command line, the Space and the tests | The live demo runs exactly the code that is tested here. | `src/ftt` is copied into the Space by a script, not installed as a package. |
| **CPU-only Docker image** | Runs anywhere with one command. The lockfile's GPU libraries (about 4 GB) are skipped and the CPU build of the same PyTorch version is used. | About 12 fps instead of the GPU's speed. |

What I would change next: a better detector (the biggest single lever, since the detector costs 25 points of HOTA for the best tracker),
pitch coordinates so the goalkeeper rule stops depending on the camera, and re-identification across camera cuts.

## Limitations

- **Two colours only.** Teams in similar colours are not handled and not measured. Striped kits are only partly handled
  (the colour histogram helps, see above, but about 1 box in 5 is still wrong on my hand-labelled amateur sample).
- **Ids restart at every camera cut.** There is no re-identification across shots.
- **Goalkeepers use image position**, so camera pans, counter-attacks and set pieces can fool the rule. It abstains on
  more than half the keepers on test (37 of 63; 7 of 16 on val).
- **Close-ups on uploaded video.** Team colours are only trusted on wide pitch views (at least 3 players, median box
  height <= 20% of the frame) and on player boxes that are themselves <= 20% of the frame height. A medium shot whose
  players sit just under that height can still slip through.
- **Detector errors are only partly caught.** On uploaded video, a "player" whose shirt colour is far from both teams
  (distance to the nearest team colour > 1.0 x the distance between the two team colours) gets no team.
  - That catches part of a black-kit referee or keeper, not all of it.
  - A club crest in a logo card was detected as a player.
- **Other frame rates and sizes.** Settings were tuned on 25 fps, 1920 px SoccerNet video. For uploads, the frame counts
  (vote window, tracker memory, colour-fit window) are scaled by the video's fps and the keeper thresholds by its width
  (`src/ftt/scale.py`); a 25 fps, 1920 px video is unchanged. Checked only in two ways:
  - *Simulation:* dropping frames of 3 val clips to 12.5 and 8.3 fps, scaling lifted team accuracy 97.2 -> 98.0 and
    95.6 -> 97.6 (label flips rose 46 -> 60 and 56 -> 75, because the vote window is shorter). Tracking scores barely moved
    (HOTA 56.1 -> 56.0 and 49.7 -> 50.4).
  - *One real 60 fps clip* (720p amateur match, no labels to score): scaling cut team-label flips 97 -> 73 and tracker ids
    81 -> 76. One clip, no ground truth, so only a direction.
- **Amateur footage and unusual kits.** Three amateur-match clips (stands view at 60 fps, an indoor dome in 4K, a drone
  view in 4K):
  - Tracking held up.
  - Team colours were unreliable: striped and similar-blue kits with hard shadows left most players in one team.
  - On the 60 fps clip the split was 74% / 26% with the mean colour and 61% / 39% with the histogram, against about
    42% / 58% in my hand-labelled boxes.
  - So the colour feature is the weak part, not the tracker.
- **Failure case seen in the demo:** in a medium shot at 38 s, a red-shirted player was drawn in the other team's colour,
  because the frame passed the pitch-view check while the colours learned on distant views did not fit. The two rules
  above now give him no team (white in the video) instead of a wrong one; the shot is still not labelled correctly.
- **Pitch coordinates and the downstream contract are deferred.** Pitch and jersey columns are null until a planned
  pitch-coordinates project fills them; a planned tactical-metrics project requires them non-null, so that contract test waits.
- **Rendering uses plain OpenCV drawing** rather than `supervision`'s annotators (fewer API changes to track).

## Future work

- Pitch coordinates (a planned follow-up project), which also fixes the goalkeeper rule's dependence on the camera.
- A vote window centred on each frame instead of one that only looks back (see "Does the vote help?").
- Re-identification across camera cuts, so a player keeps one id for the whole match.
- A colour rule that scales with each team's own colour spread (a white kit varies more than a red one).
- TrackTrack, the one Ultralytics tracker not yet benchmarked.

## Data and attribution

- **SoccerNet-Tracking** (Cioppa et al., "SoccerNet-Tracking: Multiple Object Tracking Dataset and Benchmark in Soccer
  Videos", CVPR Workshops 2022). Used under SoccerNet's terms for non-commercial research. No frames, clips or labels
  are redistributed here; only derived numbers are reported.
- **Detector:** [football-detect-serve](https://github.com/Kushagra077/football-detect-serve) (my detector project).
  Weights are referenced by path, not copied.
- **Trackers:** [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0). Metrics:
  [TrackEval](https://github.com/JonathonLuiten/TrackEval).

## License

AGPL-3.0, see [LICENSE](LICENSE). The repo uses Ultralytics' trackers, which are AGPL-3.0.
