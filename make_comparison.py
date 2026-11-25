import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "solutions_ci_demo.mp4"  # Replace with your test video filename
TARGET_FRAME = 30              # Mid-sequence frame where tracks are active
BLUR_K = 25

def apply_motion_blur(frame, k=25):
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = np.ones(k, dtype=np.float32) / k
    return cv2.filter2D(frame, -1, kernel)

def add_banner(img, text, color):
    h, w = img.shape[:2]
    banner = np.zeros((48, w, 3), dtype=np.uint8)
    cv2.putText(banner, text, (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    return np.vstack([banner, img])

model_clean = YOLO("yolo26n.pt")
model_blur = YOLO("yolo26n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
clean_vis, blur_vis = None, None

for idx in range(TARGET_FRAME + 1):
    ret, frame = cap.read()
    if not ret:
        break

    # Run tracking sequentially up to TARGET_FRAME so IDs initialize properly
    res_clean = model_clean.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, verbose=False)
    blurred_frame = apply_motion_blur(frame, BLUR_K)
    res_blur = model_blur.track(blurred_frame, persist=True, tracker="bytetrack.yaml", conf=0.15, verbose=False)

    if idx == TARGET_FRAME:
        clean_vis = res_clean[0].plot()
        blur_vis = res_blur[0].plot()

cap.release()

left = add_banner(clean_vis, "Baseline (Clean | conf=0.25 | ByteTrack)", (80, 230, 120))
right = add_banner(blur_vis, "Motion Blur k=25 (conf=0.15 | ByteTrack)", (80, 130, 255))

divider = np.full((left.shape[0], 6, 3), 255, dtype=np.uint8)
combined = np.hstack([left, divider, right])

cv2.imwrite("comparison_k25.png", combined)
print("Saved comparison_k25.png")