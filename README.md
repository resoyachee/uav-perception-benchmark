# High-Speed UAV Perception: Preliminary Motion Blur, Frame-Skip & Tracker Benchmark

A reproducible preliminary benchmark investigating real-time object detection and multi-object tracking trade-offs under simulated fast-motion stress—covering **directional motion blur**, **frame skipping**, **confidence threshold tuning**, **model scale (`yolo26n.pt` vs. `yolo26s.pt`)**, and **tracker selection (`bytetrack.yaml` vs. `botsort.yaml`)**—for high-speed UAV perception contexts.

## Context & Motivation
High-speed UAV imagery introduces challenging visual and real-time compute conditions, including motion smear during rapid maneuvers, missed detections, frame drops under compute constraints, and camera displacement. Rather than evaluating tracking solely by raw ID counts, this study measures **inference latency**, **detection volume**, and **track persistence** (average track lifespan, maximum track lifespan, and gap/reappearance events).

## Environment & Setup
* **Hardware**: NVIDIA GeForce GTX 1650 (4096 MiB VRAM)
* **Software Stack**: Python 3.13.2, PyTorch 2.11.0+cu128, Ultralytics 8.4.218
* **Models & Trackers**: `yolo26n.pt` and `yolo26s.pt` paired with `bytetrack.yaml` and `botsort.yaml` (`persist=True`)
* **Test Clip**: Reference 62-frame sample video sequence (`solutions_ci_demo.mp4`)

## Benchmark Results
All runs are exported to `benchmark_results.csv` via `benchmark_stress.py`:

| Scenario | Model | Tracker | Conf | Frames | Unique IDs | Total Box-IDs | Avg Track Len | Max Track Len | Gap Events | Avg Infer (ms) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (n + ByteTrack) | `yolo26n.pt` | `bytetrack.yaml` | 0.25 | 62 | 30 | 1318 | 43.93 | 62 | 14 | 12.10 |
| Baseline (n + BoT-SORT) | `yolo26n.pt` | `botsort.yaml` | 0.25 | 62 | 30 | 1329 | 44.30 | 62 | 19 | 12.88 |
| Baseline (s + ByteTrack) | `yolo26s.pt` | `bytetrack.yaml` | 0.25 | 62 | 20 | 970 | 48.50 | 62 | 10 | 13.65 |
| Baseline (s + BoT-SORT) | `yolo26s.pt` | `botsort.yaml` | 0.25 | 62 | 20 | 970 | 48.50 | 62 | 10 | 13.04 |
| Blur k=25 c=0.15 (n + ByteTrack) | `yolo26n.pt` | `bytetrack.yaml` | 0.15 | 62 | 11 | 193 | 17.55 | 52 | 4 | 13.46 |
| Blur k=25 c=0.15 (n + BoT-SORT) | `yolo26n.pt` | `botsort.yaml` | 0.15 | 62 | 10 | 211 | 21.10 | 52 | 4 | 12.55 |
| Blur k=25 c=0.15 (s + BoT-SORT) | `yolo26s.pt` | `botsort.yaml` | 0.15 | 62 | 15 | 384 | 25.60 | 62 | 15 | 16.65 |
| Blur k=15 + Skip=2 c=0.15 (n + BoT-SORT) | `yolo26n.pt` | `botsort.yaml` | 0.15 | 31 | 15 | 317 | 21.13 | 31 | 5 | 13.59 |
| Blur k=15 + Skip=2 c=0.15 (s + BoT-SORT) | `yolo26s.pt` | `botsort.yaml` | 0.15 | 31 | 17 | 379 | 22.29 | 31 | 5 | 13.81 |

*(Earlier `conf=0.25` vs. `conf=0.15` motion-blur and frame-skipping runs are also recorded in the project research log.)*

## Key Observations
1. **Raw ID counts alone can mask recall collapse under blur:** Heavy motion smear pushes low-contrast or fast-moving detections below threshold, reducing total box appearances even when unique ID counts appear lower.
2. **Lower confidence (`0.15`) helps preserve blurred detections:** Relaxing confidence from `0.25` to `0.15` under motion blur recovers additional box-ID appearances and lengthens active tracks without inflating unique ID counts or adding meaningful inference latency.
3. **Model capacity (`yolo26s.pt`) boosts recovery under severe blur (`k=25`):** At `conf=0.15` under `k=25` blur, `yolo26s.pt + BoT-SORT` increases `total_box_ids` to `384` (vs. `193` for `yolo26n.pt + ByteTrack` and `211` for `yolo26n.pt + BoT-SORT`), raises average track length to `25.60` frames, and reaches a `62`-frame max track—trading off higher inference time (`16.65 ms` vs. `12.55–13.46 ms`).
4. **Tighter latency vs. stronger smear resilience:** Under `k=15 + Skip=2` (`conf=0.15`), `yolo26n.pt + BoT-SORT` (`13.59 ms`) and `yolo26s.pt + BoT-SORT` (`13.81 ms`) run at nearly identical average inference times, while `yolo26s.pt` yields more detections (`379` vs. `317`) and slightly longer average tracks (`22.29` vs. `21.13` frames).

## Repository Structure
* `first_track.py` — Initial baseline tracking verification script
* `benchmark_stress.py` — Stress benchmark script (motion blur, frame skipping, confidence tuning, model/tracker sweeps, persistence metrics, and CSV export)
* `benchmark_results.csv` — Exported benchmark dataset
* `README.md` — Study overview, experimental results, and takeaways

## Next Steps
* Test on aerial/UAV-view video sequences and measure end-to-end pipeline timing (frame capture/decoding, preprocessing, inference, and tracking).
* Draft a structured preliminary technical report / benchmark note summarizing methodology, trade-offs, and future UAV perception experiments.