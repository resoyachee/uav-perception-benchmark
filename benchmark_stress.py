import csv
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4"
CSV_PATH = Path("benchmark_results.csv")


def apply_motion_blur(frame, kernel_size=15, angle=0.0):
  """Simulate directional motion blur on a frame."""
  if kernel_size <= 1:
    return frame
  kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
  kernel[kernel_size // 2, :] = 1.0
  center = (kernel_size / 2.0, kernel_size / 2.0)
  rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
  kernel = cv2.warpAffine(kernel, rot_mat, (kernel_size, kernel_size))
  s = kernel.sum()
  if s > 0:
    kernel /= s
  return cv2.filter2D(frame, -1, kernel)


def compute_persistence(id_frame_history):
  """Compute average track length, max track length, and gap/reappearance events."""
  if not id_frame_history:
    return 0.0, 0, 0

  lengths = [len(frames) for frames in id_frame_history.values()]
  avg_len = sum(lengths) / len(lengths)
  max_len = max(lengths)

  gap_events = 0
  for frames in id_frame_history.values():
    for i in range(1, len(frames)):
      if frames[i] - frames[i - 1] > 1:
        gap_events += 1

  return avg_len, max_len, gap_events


def run_benchmark(
    name,
    model_name="yolo26n.pt",
    tracker="bytetrack.yaml",
    blur_k=1,
    blur_angle=0.0,
    step=1,
    conf=0.25,
):
  model = YOLO(model_name)
  cap = cv2.VideoCapture(VIDEO_URL)
  if not cap.isOpened():
    raise RuntimeError("Could not open video source.")

  id_frame_history = {}
  id_appearances = 0
  inference_ms = []
  processed_frames = 0
  raw_frame_idx = 0

  while True:
    ok, frame = cap.read()
    if not ok:
      break

    if raw_frame_idx % step != 0:
      raw_frame_idx += 1
      continue

    if blur_k > 1:
      frame = apply_motion_blur(frame, kernel_size=blur_k, angle=blur_angle)

    results = model.track(
        frame, tracker=tracker, conf=conf, persist=True, verbose=False
    )
    r = results[0]
    inference_ms.append(r.speed["inference"])

    if r.boxes is not None and r.boxes.id is not None:
      tids = r.boxes.id.int().tolist()
      id_appearances += len(tids)
      for tid in tids:
        id_frame_history.setdefault(tid, []).append(processed_frames)

    processed_frames += 1
    raw_frame_idx += 1

  cap.release()

  unique_id_count = len(id_frame_history)
  avg_ms = sum(inference_ms) / max(len(inference_ms), 1)
  avg_len, max_len, gap_events = compute_persistence(id_frame_history)

  row = {
      "scenario": name,
      "model": model_name,
      "tracker": tracker,
      "conf": conf,
      "blur_k": blur_k,
      "step": step,
      "frames": processed_frames,
      "unique_ids": unique_id_count,
      "total_box_ids": id_appearances,
      "avg_track_len": round(avg_len, 2),
      "max_track_len": max_len,
      "gap_events": gap_events,
      "avg_infer_ms": round(avg_ms, 2),
  }

  print(
      f"[{row['scenario']}] model={row['model']} | tracker={row['tracker']} | "
      f"conf={row['conf']:.2f} | frames={row['frames']} | "
      f"unique_ids={row['unique_ids']} | total_box_ids={row['total_box_ids']} | "
      f"avg_len={row['avg_track_len']:.2f} | max_len={row['max_track_len']} | "
      f"gaps={row['gap_events']} | avg_infer={row['avg_infer_ms']:.2f} ms"
  )
  return row


def write_csv(rows, csv_path):
  fieldnames = [
      "scenario",
      "model",
      "tracker",
      "conf",
      "blur_k",
      "step",
      "frames",
      "unique_ids",
      "total_box_ids",
      "avg_track_len",
      "max_track_len",
      "gap_events",
      "avg_infer_ms",
  ]
  with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
  # Tuple format: (scenario_name, model_name, tracker, blur_k, blur_angle, step, conf)
  scenarios = [
      ("Baseline (n + ByteTrack)", "yolo26n.pt", "bytetrack.yaml", 1, 0.0, 1, 0.25),
      ("Baseline (n + BoT-SORT)", "yolo26n.pt", "botsort.yaml", 1, 0.0, 1, 0.25),
      ("Baseline (s + ByteTrack)", "yolo26s.pt", "bytetrack.yaml", 1, 0.0, 1, 0.25),
      ("Baseline (s + BoT-SORT)", "yolo26s.pt", "botsort.yaml", 1, 0.0, 1, 0.25),
      ("Blur k=25 c=0.15 (n + ByteTrack)", "yolo26n.pt", "bytetrack.yaml", 25, 15.0, 1, 0.15),
      ("Blur k=25 c=0.15 (n + BoT-SORT)", "yolo26n.pt", "botsort.yaml", 25, 15.0, 1, 0.15),
      ("Blur k=25 c=0.15 (s + BoT-SORT)", "yolo26s.pt", "botsort.yaml", 25, 15.0, 1, 0.15),
      ("Blur k=15 + Skip=2 c=0.15 (n + BoT-SORT)", "yolo26n.pt", "botsort.yaml", 15, 15.0, 2, 0.15),
      ("Blur k=15 + Skip=2 c=0.15 (s + BoT-SORT)", "yolo26s.pt", "botsort.yaml", 15, 15.0, 2, 0.15),
  ]

  all_rows = [
      run_benchmark(
          name,
          model_name=mdl,
          tracker=trk,
          blur_k=bk,
          blur_angle=ang,
          step=st,
          conf=cf,
      )
      for name, mdl, trk, bk, ang, st, cf in scenarios
  ]
  write_csv(all_rows, CSV_PATH)
  print(f"\nSaved benchmark results to {CSV_PATH.resolve()}")