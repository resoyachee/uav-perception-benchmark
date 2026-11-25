from ultralytics import YOLO

# Load a lightweight pretrained model (downloads automatically on first run)
model = YOLO("yolo26n.pt")

# Run ByteTrack on a sample video clip
results = model.track(
    source="https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4",
    tracker="bytetrack.yaml",
    persist=True,
    save=True,
)

unique_ids = set()
inference_times_ms = []

for r in results:
    inference_times_ms.append(r.speed["inference"])
    if r.boxes is not None and r.boxes.id is not None:
        for tid in r.boxes.id.int().tolist():
            unique_ids.add(tid)

avg_ms = sum(inference_times_ms) / max(len(inference_times_ms), 1)
print(f"Frames processed: {len(results)}")
print(f"Unique tracked IDs: {len(unique_ids)}")
print(f"Average inference time: {avg_ms:.2f} ms/frame")