# -*- coding: utf-8 -*-
"""
🚂 Railway Track Obstacle Detection System

This script uses your **already-trained** YOLOv8 model (`best (3).pt`) to detect obstacles
(Animal, Car, Person, Rock, Trash, Tree) that are **inside the railway track corridor only**,
and ignores everything else (e.g. trees, people, or animals standing beside the track).

## How to use this script
1. Make sure `best (3).pt` is in the same folder as this script.
2. Run this script: python railway_track_obstacle_detection_(4).py
3. It will ask you for the path to an image or video file.
4. Results will be saved as output_image.jpg or output_video.mp4.

**No retraining happens anywhere in this script** — we only run inference with your
existing weights.
"""

# Import libraries
import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (works without a display)
import matplotlib.pyplot as plt
from ultralytics import YOLO

print("Libraries ready.")


"""## Load your trained model

Place `best (3).pt` in the same folder as this script.
This script never calls `model.train()` — it only loads your existing weights for inference.
"""

MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: {MODEL_PATH} not found!")
    print("Please place the model file in the same folder as this script.")
    sys.exit(1)

model = YOLO(MODEL_PATH)
print("Model loaded. Classes:", model.names)


# Ask the user for an input file path
input_path = input("Enter the path to an image (.jpg/.png) or video (.mp4/.avi/.mov/.mkv): ").strip()

if not os.path.exists(input_path):
    print(f"ERROR: File not found: {input_path}")
    sys.exit(1)

IS_VIDEO = input_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
print(f"Input: {input_path}  |  Detected as {'VIDEO' if IS_VIDEO else 'IMAGE'}")


"""## Defining the Railway-Track ROI

Automatic (Hough-line) detection works well on a clean, direct outdoor shot of a single
track. It reliably **fails on cab-view / dashboard-camera footage that has windshield
wipers in frame** — the wipers are long, dark, near-vertical arms that periodically sweep
through the same region the track occupies, and a line detector cannot tell "wiper" from
"rail" on a frame where they happen to look similar. There's no way around this with
frame-by-frame automatic detection: it isn't a threshold you can tune away, because the
confusing object is moving through the exact search region every few frames.

For a fixed-mount camera like this (the camera doesn't move, only the wipers do), the
practical fix is a **one-time set of coordinates for this camera** — typed as plain
numbers below, not clicked. Once set, every frame after that (including every frame of
every future video from this same camera) is processed fully automatically with these
numbers, no re-entry needed.

**If your footage does NOT have wipers/dashboard in frame** (a normal outdoor photo/video),
leave `MANUAL_ROI_OVERRIDE = None` below and automatic detection will run as before.

**If it DOES have wipers/dashboard** (like the sample cab-view clip this was built
against), fill in `MANUAL_ROI_OVERRIDE` by pausing your video on a clear frame (wipers out
of the way), zooming into two horizontal strips -- one near the bottom of the frame, one
further up near the vanishing point -- and reading off the rails' pixel x-positions at
each. Four points is enough: `(left_x, y)` and `(right_x, y)` at each of the two heights.
"""


def auto_detect_roi(img, hough_threshold=80, angle_range=(20, 90), debug=False):
    """
    Detects the railway-track ROI automatically from a single frame.
    Returns a list of 4 (x, y) points (top-left, top-right, bottom-right, bottom-left),
    or None if no confident detection was possible (caller should use the fallback).
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)

    # restrict search to the lower ~75% of the frame -- rails live there, sky/rooftops don't
    mask = np.zeros_like(edges)
    mask[int(h * 0.25):, :] = 255
    edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                             minLineLength=int(h * 0.25), maxLineGap=40)
    if lines is None:
        return None

    # Reshape lines to be (N, 4) instead of (N, 1, 4) for consistent iteration
    lines = lines.reshape(-1, 4)

    candidates = []
    for x1, y1, x2, y2 in lines:
        if y2 == y1:
            continue
        angle = abs(np.degrees(np.arctan2((y2 - y1), (x2 - x1))))
        if angle_range[0] < angle <= angle_range[1]:
            candidates.append((x1, y1, x2, y2))

    if len(candidates) < 2:
        return None

    mid_x = w / 2
    left = [c for c in candidates if (c[0] + c[2]) / 2 < mid_x]
    right = [c for c in candidates if (c[0] + c[2]) / 2 >= mid_x]
    if not left or not right:
        return None

    def fit_line_extrapolate(cluster, y_top, y_bottom):
        pts = []
        for x1, y1, x2, y2 in cluster:
            pts.append((x1, y1))
            pts.append((x2, y2))
        pts = np.array(pts, dtype=np.float32)
        ys, xs = pts[:, 1], pts[:, 0]
        A = np.vstack([ys, np.ones(len(ys))]).T
        m, b = np.linalg.lstsq(A, xs, rcond=None)[0]
        return (m * y_top + b, y_top), (m * y_bottom + b, y_bottom)

    y_top = int(h * 0.38)
    y_bottom = h - 1
    left_top, left_bottom = fit_line_extrapolate(left, y_top, y_bottom)
    right_top, right_bottom = fit_line_extrapolate(right, y_top, y_bottom)

    roi = [left_top, right_top, right_bottom, left_bottom]
    roi = [(int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))) for x, y in roi]

    # --- SANITY CHECKS ---
    # These reject bad auto-detections so the fallback ROI is used instead.

    # Check 1: Left rail must stay LEFT of right rail (no crossing / bowtie)
    # roi[0]=left_top, roi[1]=right_top, roi[2]=right_bottom, roi[3]=left_bottom
    if roi[0][0] >= roi[1][0]:   # left_top.x must be < right_top.x
        return None
    if roi[3][0] >= roi[2][0]:   # left_bottom.x must be < right_bottom.x
        return None

    # Check 2: Top must be narrower than bottom (perspective vanishing point)
    top_width = abs(roi[1][0] - roi[0][0])
    bottom_width = abs(roi[2][0] - roi[3][0])
    if top_width >= bottom_width:
        return None

    # Check 3: Bottom must not be too narrow to be a real track
    if bottom_width < w * 0.03:
        return None

    # Check 4: The track center should be roughly in the middle 60% of the image
    top_center = (roi[0][0] + roi[1][0]) / 2
    bottom_center = (roi[2][0] + roi[3][0]) / 2
    avg_center = (top_center + bottom_center) / 2
    if avg_center < w * 0.2 or avg_center > w * 0.8:
        return None

    return roi


def default_fallback_roi(img, top_width_frac=0.08, bottom_width_frac=0.55, top_y_frac=0.35):
    """A generic centered trapezoid, used only if auto_detect_roi() fails outright."""
    h, w = img.shape[:2]
    cx = w / 2
    y_top = int(h * top_y_frac)
    y_bottom = h - 1
    top_half = w * top_width_frac / 2
    bottom_half = w * bottom_width_frac / 2
    return [
        (int(cx - top_half), y_top), (int(cx + top_half), y_top),
        (int(cx + bottom_half), y_bottom), (int(cx - bottom_half), y_bottom),
    ]


def draw_roi(img, roi_points, color=(255, 255, 0)):
    vis = img.copy()
    cv2.polylines(vis, [np.array(roi_points, dtype=np.int32)], True, color, 3)
    return vis


def pad_roi_horizontal(roi_points, img_w, pad_ratio=0.15):
    """
    Widens the ROI polygon outward horizontally by pad_ratio (e.g. 0.15 = 15%
    wider at both the top edge and the bottom edge, each around its own
    midpoint). The automatic detector tends to draw a slightly narrower
    trapezoid than the true track corridor, so a small safety margin here
    catches real
    obstacles near the track edge that a slightly-too-narrow
    auto-detected ROI would otherwise clip out. Raise this
    (e.g. 0.25) if obstacles near the edge are still missed;
    lower it (e.g. 0.05) if objects clearly beside the track
    start getting wrongly flagged as obstacles.
    """
    left_top, right_top, right_bottom, left_bottom = roi_points

    def widen(p_left, p_right, ratio):
        (x1, y1), (x2, y2) = p_left, p_right
        cx = (x1 + x2) / 2
        half = (x2 - x1) / 2 * (1 + ratio)
        return (cx - half, y1), (cx + half, y2)

    new_left_top, new_right_top = widen(left_top, right_top, pad_ratio)
    new_left_bottom, new_right_bottom = widen(left_bottom, right_bottom, pad_ratio)

    def clip(x):
        return int(np.clip(x, 0, img_w - 1))

    return [
        (clip(new_left_top[0]), int(new_left_top[1])),
        (clip(new_right_top[0]), int(new_right_top[1])),
        (clip(new_right_bottom[0]), int(new_right_bottom[1])),
        (clip(new_left_bottom[0]), int(new_left_bottom[1])),
    ]


# ---------------------------------------------------------
# VIDEO ROI
# ---------------------------------------------------------

VIDEO_ROI = [
    (630, 350),
    (664, 350),
    (826, 715),
    (448, 715)
]


# ---------------------------------------------------------
# READ IMAGE / VIDEO
# ---------------------------------------------------------

if IS_VIDEO:

    cap = cv2.VideoCapture(input_path)

    ok, reference_img = cap.read()

    cap.release()

    if not ok:
        raise RuntimeError("Could not read the first frame of the video.")

    # Use fixed ROI for video
    roi_points = VIDEO_ROI

    used_fallback = False

    print("Using VIDEO manual ROI.")

else:

    reference_img = cv2.imread(input_path)

    if reference_img is None:
        raise RuntimeError("Could not read the image.")

    # Use automatic ROI for images
    roi_points = auto_detect_roi(
        reference_img,
        hough_threshold=60,
        angle_range=(20, 90)
    )

    used_fallback = roi_points is None

    if used_fallback:

        print("Automatic ROI failed.")
        print("Using fallback ROI.")

        roi_points = default_fallback_roi(reference_img)

    else:

        print("Using automatic IMAGE ROI.")


# ---------------------------------------------------------
# DISPLAY ROI (saved to file instead of plt.show)
# ---------------------------------------------------------

print("Image size:", reference_img.shape[1], "x", reference_img.shape[0])
print("roi_points =", roi_points)

roi_vis = draw_roi(reference_img, roi_points)

if IS_VIDEO:
    title = "Video Manual ROI"
elif used_fallback:
    title = "Fallback Image ROI"
else:
    title = "Automatic Image ROI"

# Save the ROI visualization to a file
roi_output_path = "roi_preview.jpg"
cv2.imwrite(roi_output_path, roi_vis)
print(f"ROI preview saved as {roi_output_path} ({title})")


"""## Obstacle detection logic

**Why not just check the bottom-center point of each box (as you noticed, this missed
animals that were visually on the track)?**
A single point is fragile: if the ROI polygon's edge cuts slightly through the animal's
feet, or the bounding box extends a little past the rail, that one point can land just
outside the polygon even though most of the animal's body is clearly on the track.

**Method used here — bottom-region area overlap:**
1. Build a binary mask of the ROI polygon the same size as the frame.
2. For each detected box, take its **lower portion** (default: bottom 60% of the box —
   `bottom_fraction=0.4` means "start the region 40% down the box"), since that's the part
   of an animal/person/car that actually touches the ground/track.
3. Compute what **fraction of that lower region's area** falls inside the ROI mask.
4. If that fraction >= `overlap_thresh` (default 0.3), the object counts as **on the
   track**.

This is more forgiving than a single point (a genuinely-on-track object usually has most of
its lower body over the ROI, even if a corner of the box pokes out) while still rejecting
objects that are clearly beside the track (their lower region has almost no overlap).
"""

# Core detection + ROI-filtering function (shared by image & video)

CONF_THRESH = 0.20       # YOLO confidence threshold — see tuning notes at the end
OVERLAP_THRESH = 0.35     # fraction of an object's lower region that must be inside the ROI
BOTTOM_FRACTION = 0.4
YOLO_IMGSZ=960
YOLO_AUGMENT=True
   # 0.4 = use the bottom 60% of each box for the overlap check

# Optional: classes that should NEVER count as obstacles even if flagged inside the ROI.
# IMPORTANT: do NOT put "Tree" here -- your model uses the single "Tree" label for both
# background trees beside the track AND a fallen tree/branch lying across the track, so
# excluding it here would hide a genuine fallen-branch obstacle. The ROI overlap check
# below already handles this correctly on its own: a tree beside the track has low
# overlap (ignored), a fallen branch/tree ON the track has high overlap (flagged).
NEVER_OBSTACLE_CLASSES = set()  # e.g. {'ClassName'} only for classes that can NEVER be a real obstacle


def compute_overlap_ratio(box_xyxy, roi_mask, bottom_fraction=0.4):
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, roi_mask.shape[1]), min(y2, roi_mask.shape[0])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    y_start = int(y1 + bottom_fraction * (y2 - y1))
    region = roi_mask[y_start:y2, x1:x2]
    if region.size == 0:
        return 0.0
    return float(np.sum(region > 0)) / region.size


def detect_obstacles(frame, model, roi_points, conf_thresh=CONF_THRESH,
                      overlap_thresh=OVERLAP_THRESH, bottom_fraction=BOTTOM_FRACTION,
                      verbose=True):
    """
    Runs YOLO on `frame`, filters detections against the ROI, and draws the
    final result. Returns (annotated_frame, status_string, obstacle_found_bool).
    """
    h, w = frame.shape[:2]
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [np.array(roi_points, dtype=np.int32)], 255)

    results = model.predict(frame, conf=conf_thresh,imgsz=YOLO_IMGSZ,augment=YOLO_AUGMENT, verbose=False)[0]

    vis = frame.copy()
    cv2.polylines(vis, [np.array(roi_points, dtype=np.int32)], True, (255, 255, 0), 3)

    obstacle_found = False
    n_total, n_obstacle, n_ignored = 0, 0, 0

    if verbose:
        print(f"--- RAW YOLO DETECTIONS: {len(results.boxes)} ---")

    for box in results.boxes:
        n_total += 1
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()

        ratio = compute_overlap_ratio(xyxy, roi_mask, bottom_fraction)
        inside = (ratio >= overlap_thresh) and (cls_name not in NEVER_OBSTACLE_CLASSES)

        if verbose:
            verdict = "TRACK OBSTACLE" if inside else "ROI IGNORED"
            print(f"  [{cls_name}] conf={conf:.2f} overlap={ratio:.2f}  ->  "
                  f"YOLO DETECTED -> {verdict}")

        if inside:
            n_obstacle += 1
            obstacle_found = True
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, f"{cls_name} {conf:.2f}", (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            n_ignored += 1

    status = "OBSTACLE DETECTED" if obstacle_found else "TRACK CLEAR"
    color = (0, 0, 255) if obstacle_found else (0, 200, 0)
    cv2.putText(vis, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    if obstacle_found:
        cv2.putText(vis, "WARNING: OBSTACLE ON TRACK", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    if verbose:
        print(f"SUMMARY: total_detections={n_total}  obstacles_on_track={n_obstacle}  "
              f"ignored_outside_roi={n_ignored}  status={status}")

    return vis, status, obstacle_found


"""## Image obstacle detection

Shows the raw YOLO output first, then the final ROI-filtered result — so you can
immediately tell whether a missed object is (A) not detected by YOLO at all, or
(B) detected but filtered out by the ROI.
"""

# Run on the uploaded IMAGE
if not IS_VIDEO:
    frame = cv2.imread(input_path)

    # (A) Raw YOLO detections, no ROI filtering — for diagnostics only
    raw_results = model.predict(frame, conf=CONF_THRESH, verbose=False)[0]
    raw_vis = raw_results.plot()

    # Save raw detections image
    cv2.imwrite("raw_detections.jpg", raw_vis)
    print(f"Raw YOLO detections saved as raw_detections.jpg ({len(raw_results.boxes)} boxes, before ROI filtering)")

    # (B) Final ROI-filtered result
    annotated, status, obstacle_found = detect_obstacles(frame, model, roi_points)

    # Save final result
    cv2.imwrite("output_image.jpg", annotated)
    print(f"FINAL RESULT: {status}")
    print("Saved as output_image.jpg")
else:
    print("Skipping image processing — you provided a video. See video processing below.")


"""## Video obstacle detection

Runs YOLO + the same ROI on every frame (fixed-camera assumption — reasonable for a single
continuous clip) and writes an annotated MP4.
"""

# Run on the uploaded VIDEO
from collections import deque

SMOOTHING_WINDOW = 5  # remember the last N frames; if any had an obstacle, keep reporting
                       # OBSTACLE DETECTED even if this exact frame's detection dipped out.
                       # Raise it (e.g. 10) if flicker is still bad; lower it (e.g. 2) if you
                       # want the status to clear faster after an obstacle actually leaves.

if IS_VIDEO:
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    OUTPUT_VIDEO_PATH = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))

    recent_window = deque(maxlen=SMOOTHING_WINDOW)
    frame_idx = 0
    obstacle_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        # verbose=False here to avoid flooding output
        annotated, raw_status, raw_obstacle_found = detect_obstacles(
            frame, model, roi_points, verbose=False
        )

        recent_window.append(raw_obstacle_found)
        smoothed_obstacle_found = any(recent_window)
        smoothed_status = "OBSTACLE DETECTED" if smoothed_obstacle_found else "TRACK CLEAR"

        if smoothed_obstacle_found and not raw_obstacle_found:
            # this single frame looked clear, but an obstacle was seen very recently --
            # override the on-frame text so the video doesn't flicker to "clear" for
            # one missed frame while the obstacle is still almost certainly there.
            cv2.rectangle(annotated, (10, 10), (560, 95), (0, 0, 0), -1)
            cv2.putText(annotated, smoothed_status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            cv2.putText(annotated, "WARNING: OBSTACLE ON TRACK", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if smoothed_obstacle_found:
            obstacle_frames += 1
        writer.write(annotated)

        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"Processed frame {frame_idx}/{total_frames}  |  "
                  f"raw={raw_status}  smoothed={smoothed_status}")

    cap.release()
    writer.release()
    print(f"\nDone. {obstacle_frames}/{frame_idx} frames had an obstacle on the track "
          f"(after smoothing).")
    print(f"Saved annotated video to {OUTPUT_VIDEO_PATH}")
else:
    print("Skipping video processing — you provided an image. See image processing above.")


# Final output summary
print("\n" + "=" * 60)
print("PROCESSING COMPLETE")
print("=" * 60)
if IS_VIDEO:
    print(f"Output video: output_video.mp4")
    print(f"ROI preview:  roi_preview.jpg")
else:
    print(f"Output image:     output_image.jpg")
    print(f"Raw detections:   raw_detections.jpg")
    print(f"ROI preview:      roi_preview.jpg")


"""## Tuning guide

- **`CONF_THRESH`** (default `0.20`): YOLO's minimum confidence to keep a detection
  at all. Lower it (e.g. `0.15`) if real obstacles are being missed by YOLO itself — video
  frames are usually lower quality than still photos (compression, motion blur), so this
  often needs to be lower for video than what worked for your test image. Check the
  raw_detections.jpg output — if the object isn't boxed there, it's a YOLO recall issue,
  not an ROI issue. Raise it (e.g. `0.5`) if you're getting false positives.
- **`ROI_PADDING`** (default `0.15`): widens the auto-detected/fallback ROI by
  this fraction. The automatic detector tends to draw a slightly narrower trapezoid than
  the true track, so this compensates. Raise it (e.g. `0.25`) if obstacles near the track
  edge are being missed in video; lower it if objects clearly beside the track start
  getting wrongly flagged.
- **`OVERLAP_THRESH`** (default `0.35`): how much of an object's lower body must sit
  inside the ROI to count as "on the track". Lower it (e.g. `0.15–0.2`) if genuinely-on-track
  objects near the ROI edge are being ignored; raise it (e.g. `0.5`) if objects merely
  brushing the ROI boundary are being wrongly flagged as obstacles.
- **`BOTTOM_FRACTION`** (default `0.4`): how much of the top of each box to *exclude*
  before measuring overlap. `0.4` means only the bottom 60% of the box is checked (this is
  the part of an animal/person/car that's actually on the ground). Set closer to `0.0` to
  use almost the whole box, or closer to `0.8` to use only the very bottom sliver.
- **`NEVER_OBSTACLE_CLASSES`** (default empty): only add a class here if it can
  *never physically be a real obstacle no matter where it is* (there's no such class in
  your set — even `Tree` doesn't qualify, since your model uses `Tree` for both background
  trees and fallen branches on the track). Leave this empty and let the ROI overlap check
  do the filtering; that's what correctly distinguishes "tree beside the track" from
  "tree/branch on the track".

## Full pipeline, in simple terms

```
Image / Video frame
      │
      ▼
YOLO detection (your best.pt, unchanged)      → every object + class + confidence + box
      │
      ▼
Railway Track ROI (auto suggestion or your manual click points)
      │
      ▼
Spatial filtering (bottom-portion-of-box vs ROI mask overlap ratio)
      │
      ├── overlap < threshold  →  IGNORED (outside track)
      │
      └── overlap ≥ threshold  →  OBSTACLE INSIDE TRACK
                                        │
                                        ▼
                              "OBSTACLE DETECTED" +
                            "WARNING: OBSTACLE ON TRACK"
```

"""