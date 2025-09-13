# 🧠 NeuroShield – Parkinson’s Gait Analysis

NeuroShield is an **AI-powered healthcare tool** that analyzes gait patterns from videos to assist in **Parkinson’s disease monitoring**.  
Using **YOLOv8 pose estimation**, it extracts stride features, detects asymmetry, and generates insights with visual graphs.

---

## ✨ Inspiration
Parkinson’s patients often experience gait disturbances that go unnoticed until advanced stages.  
We wanted to build a tool that empowers **early detection, monitoring, and patient support** using AI for **social good in healthcare**.

---

## 🚀 What It Does
- 📹 Upload a gait video of a patient.  
- 🦵 Detects keypoints (ankles, head, stride movements).  
- 📊 Calculates **stride length, variability, and asymmetry**.  
- 📉 Classifies gait as **Stable / Slightly Unstable / Unstable**.  
- 📈 Generates graphs & recommendations for clinicians/patients.  

---

## 🛠 Tech Stack
- **Python 3.10**
- **Streamlit** (frontend & UI)
- **YOLOv8 Pose (Ultralytics)** for keypoint detection
- **OpenCV & NumPy** for video/frame processing
- **Matplotlib & Pandas** for metrics and graphs
- **Scipy** for stride signal smoothing
- *(Optional: Demographics dataset `demographics.xls` for patient metadata)*

