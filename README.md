# Real-Time Stadium Violence Detection and Density-Aware Risk Fusion

## 🎯 Project Overview
**Real-Time Stadium Violence Detection and Density-Aware Risk Fusion** is an AI-powered real-time surveillance system designed to detect violent objects (such as bats), estimate crowd density, and compute a global stadium risk level using a multi-module pipeline.

This system integrates:
- YOLOv10 custom object detection
- Crowd density estimation with YOLO-based analysis
- A risk fusion algorithm combining density + weapon detection
- Annotated video generation
- CSV logging
- A visual dashboard showing global risk status

---

## 🚀 Features
- Real-time detection of dangerous objects (bat)
- Crowd density estimation in LOW / MEDIUM / HIGH zones
- Global risk classification: SAFE, LOW, MEDIUM, HIGH, CRITICAL
- Automatic risk evaluation
- CSV logging of all frame metrics
- Annotated video output
- Modular design (alerts, detection, counting, preprocessing)
- Fully compatible with **uv** package manager

---


## ▶️ How to Run the System (Using uv)

This project is fully compatible with the **uv** package manager, allowing fast and isolated dependency management without needing a `requirements.txt`.


---

## 1️⃣ Create & Sync Environment with uv

Inside the project folder:

```bash
uv venv
uv sync
```
If any instalation is requiered use
```bash
uv pip install ultralytics 
opencv-python 
numpy pandas  
matplotlib

```
## 📥 2️⃣ Download Required Files (Dataset + Model + Test Videos)

Before running the system, **you MUST download the dataset, the trained model, and the sample test videos.**

All necessary files are available in the following OneDrive folder:

🔗 **Download from OneDrive:**  
https://winliveudlap-my.sharepoint.com/:f:/r/personal/jesus_navacn_udlap_mx/Documents/Datasets%20and%20Weights?csf=1&web=1&e=NMapuU
Inside the folder you will find:

- `dataset/` → YOLOv10 dataset for retraining  
- `train` → trained YOLOv10 model for bat detection  
- `videos/` → test videos to run the pipeline  

---

## 🎬 3️⃣ Run the Real-Time Pipeline

### ▶️ Run with a video file

To run the system with a video, you must first set the correct path inside `main_pipeline.py`.

Locate the following line:

\`\`\`python
video_path = os.path.abspath("videos/prueba1_restaurado.mp4")
\`\`\`

Replace `"prueba1_restaurado.mp4"` with the name of the video you want to test.

Example:

```python
video_path = os.path.abspath("videos/test_video_02.mp4")
```

After selecting the correct video path, run:

```bash
uv run python main_pipeline.py
```

The system will automatically:

- Detect bats  
- Count people  
- Estimate density  
- Compute GLOBAL RISK  
- Save CSV logs  
- Generate an annotated output video  

---


