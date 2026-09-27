# 🚂 Railway Track Obstacle Detection System

AI-based Railway Track Obstacle Detection using YOLOv8 and Streamlit.

This application uses a **pre-trained YOLO model** (`best (3).pt`) to detect obstacles
(Animal, Car, Person, Rock, Trash, Tree) that are located **on the railway track**.

Objects detected **outside** the railway track ROI (Region of Interest) are automatically ignored.

---

## 📁 Project Files

```
files/
├── app.py              ← Main Streamlit application (run this)
├── detector.py         ← Detection logic (helper functions)
├── best (3).pt         ← Pre-trained YOLO model (DO NOT delete)
├── requirements.txt    ← Python packages needed
└── README.md           ← This file
```

---

## 🚀 How to Run

### Step 1: Open the project folder

Open the folder where all the project files are located.

### Step 2: Open Command Prompt (or Terminal)

- On **Windows**: Hold `Shift` + Right-click inside the folder → "Open PowerShell window here"
- Or open Command Prompt and navigate to the folder:
  ```
  cd C:\Users\User\Downloads\files
  ```

### Step 3: Install the required packages

Run this command:

```bash
pip install -r requirements.txt
```

This will install Streamlit, YOLO, OpenCV, and other packages the app needs.

> **Note:** If you get a permission error, try: `pip install --user -r requirements.txt`

### Step 4: Make sure these files are together in the same folder

```
app.py
detector.py
best (3).pt
requirements.txt
```

All of these files must be in the **same folder**. The app will not work if `best (3).pt` is missing.

### Step 5: Run the Streamlit app

```bash
streamlit run app.py
```

### Step 6: Open the Streamlit URL

After running the command, you will see a message like:

```
Local URL: http://localhost:8501
```

Open that URL in your web browser (Chrome, Firefox, Edge, etc.).

---

## 🖼️ How to Use — Image Mode

1. Select **📷 Image** in the app
2. Upload a `.jpg`, `.jpeg`, or `.png` image of a railway track
3. The app will:
   - Detect objects using YOLO
   - Calculate the railway track ROI
   - Check which objects are **on the track**
   - Show the result: **OBSTACLE DETECTED** or **TRACK CLEAR**
4. You can download the annotated result image

---

## 🎥 How to Use — Video Mode

1. Select **🎥 Video** in the app
2. Upload a `.mp4`, `.avi`, `.mov`, or `.mkv` video
3. Click **Start Processing Video**
4. The app will:
   - Process each frame with YOLO
   - Apply ROI filtering on every frame
   - Use smoothing to reduce status flicker
   - Show a live preview while processing
5. After processing, you can watch and download the annotated video

---

## ⚙️ Settings (Sidebar)

You can adjust these settings in the sidebar:

| Setting | Default | What it does |
|---------|---------|-------------|
| **Confidence Threshold** | 0.20 | Minimum YOLO confidence to keep a detection. Lower = more detections. |
| **ROI Overlap Threshold** | 0.35 | How much of an object's bottom region must overlap with the track ROI. Lower = more lenient. |

---

## 🔍 How the Detection Works

```
Image / Video frame
      │
      ▼
YOLO detection (best.pt)  →  detects all objects + class + confidence + bounding box
      │
      ▼
Railway Track ROI (automatic for images, fixed for videos)
      │
      ▼
Spatial filtering (checks if the BOTTOM portion of the bounding box overlaps the ROI)
      │
      ├── overlap < threshold  →  IGNORED (object is outside the track)
      │
      └── overlap ≥ threshold  →  OBSTACLE ON TRACK  →  "OBSTACLE DETECTED"
```

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| `best (3).pt was not found` | Make sure the model file is in the same folder as `app.py` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Video won't play in browser | Use the Download button to save and play locally |
| App is slow on video | Video processing is frame-by-frame — larger videos take longer |

---

## 📝 Important Notes

- The model is **already trained**. This app only runs **inference** (prediction).
- The app does **NOT** retrain or modify the model.
- The ROI filtering ensures only objects **on the railway track** are flagged as obstacles.
