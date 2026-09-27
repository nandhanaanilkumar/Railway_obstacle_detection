# detector.py
# -----------
# This file contains the detection logic for the Railway Track Obstacle Detection System.
# It is ported directly from the original Colab notebook.
# Nothing here trains a model — it only runs inference with the existing best.pt weights.

import cv2
import numpy as np
from collections import deque


# ──────────────────────────────────────────────
# DEFAULT SETTINGS (same as the original notebook)
# ──────────────────────────────────────────────

CONF_THRESH = 0.20          # YOLO minimum confidence to keep a detection
OVERLAP_THRESH = 0.25       # fraction of object's lower region that must be inside the ROI
BOTTOM_FRACTION = 0.4                      # 0.4 = use the bottom 60% of each bounding box for the overlap check
YOLO_IMGSZ = 960            # image size passed to YOLO for inference
YOLO_AUGMENT = False        # test-time augmentation
SMOOTHING_WINDOW = 5        # number of past frames to remember for video smoothing

# Classes that should NEVER count as obstacles (empty by default — see tuning guide in Colab)
NEVER_OBSTACLE_CLASSES = set()

# Fixed ROI coordinates for video (from the original notebook — do NOT change these)
VIDEO_ROI = [
    (630, 350),
    (664, 350),
    (826, 715),
    (448, 715),
]


# ──────────────────────────────────────────────
# ROI FUNCTIONS
# ──────────────────────────────────────────────

def auto_detect_roi(img, hough_threshold=80, angle_range=(20, 90)):
    """
    Tries to automatically detect the railway-track ROI from a single image
    using edge detection and Hough lines.

    Returns a list of 4 (x, y) points if successful, or None if it fails
    (the caller should then use the fallback ROI).
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)

    # Only look at the lower 75% of the frame (rails are there, not in the sky)
    mask = np.zeros_like(edges)
    mask[int(h * 0.25):, :] = 255
    edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                             minLineLength=int(h * 0.25), maxLineGap=40)
    if lines is None:
        return None

    # Reshape lines to (N, 4) for easy iteration
    lines = lines.reshape(-1, 4)

    # Keep only lines that have a steep-ish angle (rails are roughly vertical)
    candidates = []
    for x1, y1, x2, y2 in lines:
        if y2 == y1:
            continue
        angle = abs(np.degrees(np.arctan2((y2 - y1), (x2 - x1))))
        if angle_range[0] < angle <= angle_range[1]:
            candidates.append((x1, y1, x2, y2))

    if len(candidates) < 2:
        return None

    # Split candidates into left-of-center and right-of-center groups
    mid_x = w / 2
    left = [c for c in candidates if (c[0] + c[2]) / 2 < mid_x]
    right = [c for c in candidates if (c[0] + c[2]) / 2 >= mid_x]
    if not left or not right:
        return None

    def fit_line_extrapolate(cluster, y_top, y_bottom):
        """Fit a line through a cluster of segments and extrapolate to y_top and y_bottom."""
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
    """
    Creates a generic centered trapezoid ROI.
    Used only when auto_detect_roi() fails.
    """
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


def get_image_roi(img):
    """
    For images: try automatic ROI detection first, fall back to default trapezoid.
    Returns (roi_points, used_fallback_bool).
    """
    roi_points = auto_detect_roi(img, hough_threshold=60, angle_range=(20, 90))
    if roi_points is None:
        return default_fallback_roi(img), True
    return roi_points, False


def get_video_roi():
    """
    For videos: always return the fixed manual ROI coordinates.
    """
    return VIDEO_ROI


# ──────────────────────────────────────────────
# OVERLAP CALCULATION (the core spatial filtering)
# ──────────────────────────────────────────────

def compute_overlap_ratio(box_xyxy, roi_mask, bottom_fraction=BOTTOM_FRACTION):
    """
    Computes what fraction of the BOTTOM portion of a bounding box
    falls inside the ROI mask.

    This is better than checking a single point because it handles
    objects near the ROI edge more accurately.

    Args:
        box_xyxy: bounding box as [x1, y1, x2, y2]
        roi_mask: binary mask (same size as the frame) with the ROI filled in white
        bottom_fraction: how much of the top of the box to skip (0.4 = use bottom 60%)

    Returns:
        A float between 0.0 and 1.0 (fraction of the bottom region inside the ROI)
    """
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, roi_mask.shape[1]), min(y2, roi_mask.shape[0])
    if x2 <= x1 or y2 <= y1:
        return 0.0

    # Start from bottom_fraction down the box (skip the top part)
    y_start = int(y1 + bottom_fraction * (y2 - y1))
    region = roi_mask[y_start:y2, x1:x2]
    if region.size == 0:
        return 0.0

    return float(np.sum(region > 0)) / region.size


# ──────────────────────────────────────────────
# MAIN DETECTION FUNCTION (used for both images and videos)
# ──────────────────────────────────────────────

def detect_obstacles(frame, model, roi_points,
                     conf_thresh=CONF_THRESH,
                     overlap_thresh=OVERLAP_THRESH,
                     bottom_fraction=BOTTOM_FRACTION):
    """
    Runs YOLO on a single frame, filters detections against the ROI,
    and draws the annotated result.

    Returns:
        annotated_frame: the frame with boxes, ROI, and status text drawn on it
        status: "OBSTACLE DETECTED" or "TRACK CLEAR"
        obstacle_found: True/False
        on_track_list: list of dicts with info about objects ON the track
        ignored_count: number of objects detected but outside the track ROI
    """
    h, w = frame.shape[:2]

    # Create a binary mask from the ROI polygon
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [np.array(roi_points, dtype=np.int32)], 255)

    # Run YOLO inference (NO training!)
    results = model.predict(
        frame,
        conf=conf_thresh,
        iou=0.40,
        imgsz=YOLO_IMGSZ,
        augment=YOLO_AUGMENT,
        verbose=False
    )[0]

    # Start drawing on a copy of the frame
    vis = frame.copy()

    # Draw the ROI polygon in yellow
    cv2.polylines(vis, [np.array(roi_points, dtype=np.int32)], True, (255, 255, 0), 3)

    obstacle_found = False
    on_track_list = []   # objects that ARE on the track
    ignored_count = 0    # objects detected but OUTSIDE the track

    # Check each detected object
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()

        # Calculate how much of the object's bottom region overlaps with the ROI
        ratio = compute_overlap_ratio(xyxy, roi_mask, bottom_fraction)
        inside = (ratio >= overlap_thresh) and (cls_name not in NEVER_OBSTACLE_CLASSES)

        if inside:
            # This object IS on the railway track — it's an obstacle!
            obstacle_found = True
            on_track_list.append({
                "class": cls_name,
                "confidence": round(conf, 2),
                "overlap": round(ratio, 2),
            })

            # Draw a red bounding box and label
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, f"{cls_name} {conf:.2f}", (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            # This object is NOT on the track — ignore it
            ignored_count += 1

    # Draw the overall status text on the frame
    status = "OBSTACLE DETECTED" if obstacle_found else "TRACK CLEAR"
    color = (0, 0, 255) if obstacle_found else (0, 200, 0)
    cv2.putText(vis, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    if obstacle_found:
        cv2.putText(vis, "WARNING: OBSTACLE ON TRACK", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return vis, status, obstacle_found, on_track_list, ignored_count


# ──────────────────────────────────────────────
# VIDEO PROCESSING (with smoothing to reduce flicker)
# ──────────────────────────────────────────────

def process_video(input_path, output_path, model, roi_points,
                  conf_thresh=CONF_THRESH,
                  overlap_thresh=OVERLAP_THRESH,
                  progress_callback=None,
                  frame_callback=None):
    """
    Processes a video file frame-by-frame:
    1. Runs YOLO + ROI filtering on each frame
    2. Applies smoothing to reduce status flicker
    3. Writes an annotated output video

    Args:
        input_path: path to the input video file
        output_path: path to save the annotated output video
        model: loaded YOLO model
        roi_points: list of 4 (x, y) tuples defining the track ROI
        conf_thresh: YOLO confidence threshold
        overlap_thresh: ROI overlap threshold
        progress_callback: function(current_frame, total_frames) called after each frame
        frame_callback: function(annotated_frame) called with each processed frame for live preview

    Returns:
        (total_frames, obstacle_frames) — how many frames total and how many had obstacles
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Smoothing: remember the last N frames to reduce flicker
    recent_window = deque(maxlen=SMOOTHING_WINDOW)
    frame_idx = 0
    obstacle_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        # Run detection on this frame
        annotated, raw_status, raw_obstacle_found, _, _ = detect_obstacles(
            frame, model, roi_points,
            conf_thresh=conf_thresh,
            overlap_thresh=overlap_thresh
        )

        # Smoothing: if any recent frame had an obstacle, keep showing OBSTACLE DETECTED
        recent_window.append(raw_obstacle_found)
        smoothed_obstacle_found = any(recent_window)

        if smoothed_obstacle_found and not raw_obstacle_found:
            # This frame looks clear, but an obstacle was seen very recently —
            # override the text so the video doesn't flicker to "clear" briefly
            smoothed_status = "OBSTACLE DETECTED"
            cv2.rectangle(annotated, (10, 10), (560, 95), (0, 0, 0), -1)
            cv2.putText(annotated, smoothed_status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            cv2.putText(annotated, "WARNING: OBSTACLE ON TRACK", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if smoothed_obstacle_found:
            obstacle_frames += 1

        writer.write(annotated)

        # Send live preview frame (every 3rd frame to reduce overhead)
        if frame_callback and (frame_idx % 3 == 0 or frame_idx == total_frames):
            # Convert BGR to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_callback(frame_rgb, frame_idx, total_frames)

        # Update progress bar
        if progress_callback:
            progress_callback(frame_idx, total_frames)

    cap.release()
    writer.release()

    return total_frames, obstacle_frames
