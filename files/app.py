# app.py
# -----
# 🚂 Railway Track Obstacle Detection — Streamlit Application
#
# This is the main file. Run it with:
#     streamlit run app.py
#
# Make sure "best (3).pt" is in the same folder as this file.

import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from PIL import Image
from ultralytics import YOLO

# Import our detection helper functions
from detector import (
    get_image_roi,
    get_video_roi,
    detect_obstacles,
    process_video,
    CONF_THRESH,
    OVERLAP_THRESH,
)


# ──────────────────────────────────────────────
# 1. PAGE CONFIGURATION
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Railway Track Obstacle Detection",
    page_icon="🚂",
    layout="wide",
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #F5F7FA;
    }

    /* Header styling */
    .main-title {
        color: #1565C0;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
        text-align: center;
    }
    .sub-title {
        color: #1F2937;
        font-size: 1.1rem;
        font-weight: 400;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .description {
        color: #6B7280;
        font-size: 0.95rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* Result cards */
    .result-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .obstacle-detected {
        border-left: 5px solid #DC2626;
    }
    .track-clear {
        border-left: 5px solid #16A34A;
    }

    /* Status banners */
    .status-obstacle {
        background: linear-gradient(90deg, #DC2626 0%, #F87171 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        margin: 0.5rem 0;
    }
    .status-clear {
        background: linear-gradient(90deg, #16A34A 0%, #4ADE80 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        margin: 0.5rem 0;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1E293B;
    }
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown span {
        color: #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] label {
        color: #E2E8F0 !important;
    }

    /* Info metric cards */
    .metric-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1565C0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6B7280;
        margin-top: 0.2rem;
    }

    /* Divider */
    .styled-divider {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #1565C0, #FF9800, #1565C0);
        margin: 1.5rem 0;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 2. LOAD THE YOLO MODEL
# ──────────────────────────────────────────────

MODEL_PATH = "files/best.pt"


@st.cache_resource
def load_model():
    """
    Loads the trained YOLO model from best.pt.
    This function is cached so the model is only loaded once,
    even if Streamlit re-runs the script.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    model = YOLO(MODEL_PATH)
    return model


# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    st.error(
        f"❌ **{MODEL_PATH}** was not found.\n\n"
        "Please place the model file in the same folder as `app.py`."
    )
    st.stop()

# Load the model
model = load_model()
if model is None:
    st.error("❌ Failed to load the model. Please check that the model file is valid.")
    st.stop()


# ──────────────────────────────────────────────
# 3. SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🚂 Railway Detector")
    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    # Model Information
    st.markdown("### 📋 Model Information")
    st.markdown(f"""
    - **Model:** YOLOv8
    - **Weights:** best.pt
    - **Mode:** Inference only
    - **Classes:** {', '.join(model.names.values())}
    """)

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    # Detection Settings
    st.markdown("### ⚙️ Detection Settings")

    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=CONF_THRESH,
        step=0.05,
        help="Minimum YOLO confidence to keep a detection. Lower = more detections (but more false positives)."
    )

    overlap_thresh = st.slider(
        "ROI Overlap Threshold",
        min_value=0.10,
        max_value=1.00,
        value=OVERLAP_THRESH,
        step=0.05,
        help="How much of an object's bottom region must be inside the track ROI. Lower = more lenient."
    )

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    st.markdown("### ℹ️ How it works")
    st.markdown("""
    1. YOLO detects objects in the frame
    2. A **Railway Track ROI** (Region of Interest) is defined
    3. Only objects whose **lower body overlaps** the track ROI are flagged as obstacles
    4. Objects outside the track are **ignored**
    """)


# ──────────────────────────────────────────────
# 4. MAIN HEADER
# ──────────────────────────────────────────────

st.markdown('<h1 class="main-title">🚂 Railway Track Obstacle Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-based Railway Track Obstacle Detection</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="description">Upload an image or video and the system will detect '
    'possible obstacles located on the railway track.</p>',
    unsafe_allow_html=True
)
st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 5. INPUT MODE SELECTION
# ──────────────────────────────────────────────

mode = st.radio(
    "Select input type:",
    ["📷 Image", "🎥 Video"],
    horizontal=True,
)


# ══════════════════════════════════════════════
# 6. IMAGE MODE
# ══════════════════════════════════════════════

if mode == "📷 Image":
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        help="Upload a railway track image to detect obstacles."
    )

    if uploaded_image is not None:
        try:
            # Read the uploaded image
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if original_img is None:
                st.error("❌ Could not read the image. It may be corrupted or in an unsupported format.")
                st.stop()

            # Show the original image
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📸 Original Image")
                # Convert BGR to RGB for display
                original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                st.image(original_rgb, width="stretch")

            # Calculate the railway-track ROI for this image
            roi_points, used_fallback = get_image_roi(original_img)

            if used_fallback:
                st.info("ℹ️ Automatic ROI detection failed. Using fallback ROI (centered trapezoid).")
            else:
                st.success("✅ Automatic ROI detection successful.")

            # Run the detection with ROI filtering
            with st.spinner("🔍 Detecting obstacles..."):
                annotated_img, status, obstacle_found, on_track_list, ignored_count = detect_obstacles(
                    original_img, model, roi_points,
                    conf_thresh=conf_thresh,
                    overlap_thresh=overlap_thresh,
                )

            # Show the annotated result
            with col2:
                st.markdown("#### 🔍 Detection Result")
                annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, width="stretch")

            st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

            # ── Status banner ──
            if obstacle_found:
                st.markdown(
                    '<div class="status-obstacle">🚨 OBSTACLE DETECTED</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="status-clear">✅ TRACK CLEAR</div>',
                    unsafe_allow_html=True
                )

            # ── Detection details ──
            st.markdown("### 📊 Detection Result")

            # Metric cards
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{len(on_track_list)}</div>'
                    f'<div class="metric-label">Obstacles on Track</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with m2:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{ignored_count}</div>'
                    f'<div class="metric-label">Ignored (Outside Track)</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with m3:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{len(on_track_list) + ignored_count}</div>'
                    f'<div class="metric-label">Total Detections</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Table of obstacles on track
            if on_track_list:
                st.markdown("#### 🎯 Obstacles on the Railway Track")
                # Build a simple table
                table_data = {
                    "Object": [obj["class"] for obj in on_track_list],
                    "Confidence": [obj["confidence"] for obj in on_track_list],
                    "ROI Overlap": [obj["overlap"] for obj in on_track_list],
                }
                st.table(table_data)

            if ignored_count > 0:
                st.markdown(
                    f"ℹ️ **{ignored_count}** object(s) were detected by YOLO but ignored "
                    f"because they are **outside the railway track ROI**."
                )

            # ── Download button ──
            st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

            # Encode the annotated image to bytes for download
            success, img_encoded = cv2.imencode(".jpg", annotated_img)
            if success:
                st.download_button(
                    label="📥 Download Result Image",
                    data=img_encoded.tobytes(),
                    file_name="obstacle_detection_result.jpg",
                    mime="image/jpeg",
                )

        except Exception as e:
            st.error(f"❌ An error occurred while processing the image: {str(e)}")


# ══════════════════════════════════════════════
# 7. VIDEO MODE
# ══════════════════════════════════════════════

elif mode == "🎥 Video":
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a railway track video to detect obstacles."
    )

    if uploaded_video is not None:
        try:
            # Save the uploaded video to a temporary file (OpenCV needs a file path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
                tmp_input.write(uploaded_video.read())
                input_video_path = tmp_input.name

            # Try reading the first frame to verify the video is valid
            test_cap = cv2.VideoCapture(input_video_path)
            ok, first_frame = test_cap.read()
            total_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            test_cap.release()

            if not ok:
                st.error("❌ Could not read the video. It may be corrupted or in an unsupported format.")
                st.stop()

            st.success(f"✅ Video loaded successfully — **{total_frames}** frames detected.")

            # Show the first frame
            st.markdown("#### 📹 First Frame Preview")
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            st.image(first_frame_rgb, width="stretch", caption="First frame of the video")

            # Use the fixed video ROI
            roi_points = get_video_roi()
            st.info("ℹ️ Using fixed video ROI (pre-configured for standard railway camera angle).")

            # Process button
            if st.button("🚀 Start Processing Video", type="primary"):
                st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
                st.markdown("### ⏳ Processing Video...")

                # Create a temporary output file
                output_video_path = tempfile.mktemp(suffix=".mp4")

                # Set up the progress bar and status text
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Set up live preview
                preview_placeholder = st.empty()

                def update_progress(current, total):
                    """Called after each frame to update the progress bar."""
                    if total > 0:
                        progress_bar.progress(current / total)

                def update_preview(frame_rgb, current, total):
                    """Called periodically to show a live preview of the processed frame."""
                    status_text.markdown(f"**Processing frame {current} / {total}**")
                    preview_placeholder.image(frame_rgb, caption=f"Frame {current}/{total}", width="stretch")

                # Process the video
                total_processed, obstacle_frame_count = process_video(
                    input_path=input_video_path,
                    output_path=output_video_path,
                    model=model,
                    roi_points=roi_points,
                    conf_thresh=conf_thresh,
                    overlap_thresh=overlap_thresh,
                    progress_callback=update_progress,
                    frame_callback=update_preview,
                )

                # Done!
                progress_bar.progress(1.0)
                status_text.markdown("**✅ Processing complete!**")

                st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

                # Show summary
                st.markdown("### 📊 Video Processing Summary")

                s1, s2, s3 = st.columns(3)
                with s1:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value">{total_processed}</div>'
                        f'<div class="metric-label">Total Frames</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with s2:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value">{obstacle_frame_count}</div>'
                        f'<div class="metric-label">Frames with Obstacles</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with s3:
                    pct = round(100 * obstacle_frame_count / max(total_processed, 1), 1)
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value">{pct}%</div>'
                        f'<div class="metric-label">Obstacle Percentage</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                

                if os.path.exists(output_video_path):
                    # Try to display the video directly
                    with open(output_video_path, "rb") as video_file:
                            video_bytes = video_file.read()

                        
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=video_bytes,
                        file_name="obstacle_detection_result.mp4",
                        mime="video/mp4",
                    )
                else:
                    st.error("❌ Output video file was not created. Something went wrong during processing.")

                # Clean up temporary files
                try:
                    os.unlink(input_video_path)
                    if os.path.exists(output_video_path):
                        os.unlink(output_video_path)
                except Exception:
                    pass  # Not critical if cleanup fails

        except Exception as e:
            st.error(f"❌ An error occurred while processing the video: {str(e)}")


# ──────────────────────────────────────────────
# 8. FOOTER
# ──────────────────────────────────────────────

st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color:#9CA3AF; font-size:0.85rem;">'
    '🚂 Railway Track Obstacle Detection System — Built with YOLOv8 & Streamlit'
    '</p>',
    unsafe_allow_html=True
)
