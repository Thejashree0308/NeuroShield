import os
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
model = YOLO("yolov8n-pose.pt")
try:
    df = pd.read_excel("demographics.xls")
    healthy_ref = df[df['parkinson_stage'] == 0].mean(numeric_only=True)
    pd_ref = df[df['parkinson_stage'] > 0].mean(numeric_only=True)
    dataset_loaded = True
except Exception as e:
    dataset_loaded = False
    healthy_ref, pd_ref = None, None
st.title("🧠 NeuroShield – Parkinson’s Gait Analysis")

video_file = st.file_uploader("Upload gait video", type=["mp4", "avi", "mov", "mkv", "webm"])

STEP_THRESHOLD_PX = 5
ADULT_HEIGHT_M = 1.65
FRAME_SKIP = 3

if video_file:
    file_extension = os.path.splitext(video_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tfile:
        tfile.write(video_file.read())
        temp_path = tfile.name

    st.video(temp_path)

    with st.spinner("Analyzing gait video..."):
        results = model.predict(temp_path, imgsz=320, stream=True)

        left_ankle, right_ankle = [], []
        person_heights_px = []

        for i, r in enumerate(results):
            if i % FRAME_SKIP != 0:
                continue
            if r.keypoints is not None:
                kpts = r.keypoints.xy.cpu().numpy()
                for person in kpts:
                    l = person[15]
                    r_ = person[16]
                    left_ankle.append(l)
                    right_ankle.append(r_)
                    head = person[0]
                    foot_mid = (l + r_) / 2
                    person_heights_px.append(np.linalg.norm(foot_mid - head))

        left_ankle = np.array(left_ankle)
        right_ankle = np.array(right_ankle)

        if len(person_heights_px) > 0:
            avg_height_px = np.mean(person_heights_px)
            PIXEL_TO_METER = ADULT_HEIGHT_M / avg_height_px
        else:
            PIXEL_TO_METER = 0.02

        if len(left_ankle) > 5:
            left_y_smooth = uniform_filter1d(left_ankle[:, 1], size=5)
            right_y_smooth = uniform_filter1d(right_ankle[:, 1], size=5)
        else:
            left_y_smooth = left_ankle[:, 1] if len(left_ankle) > 0 else np.array([])
            right_y_smooth = right_ankle[:, 1] if len(right_ankle) > 0 else np.array([])

        def compute_strides(y_smooth):
            if len(y_smooth) < 5:
                return np.array([])
            peaks, _ = find_peaks(y_smooth, distance=5)
            strides = []
            for i in range(1, len(peaks)):
                dist = abs(y_smooth[peaks[i]] - y_smooth[peaks[i-1]])
                if dist > STEP_THRESHOLD_PX:
                    strides.append(dist * PIXEL_TO_METER)
            return np.array(strides)

        left_strides = compute_strides(left_y_smooth)
        right_strides = compute_strides(right_y_smooth)
        all_strides = np.concatenate([left_strides, right_strides]) if len(left_strides) > 0 and len(right_strides) > 0 else np.array([])
        asymmetry = abs(np.mean(left_strides) - np.mean(right_strides)) if len(left_strides) > 0 and len(right_strides) > 0 else 0

        if len(all_strides) > 0:
            mean_stride = np.mean(all_strides)
            stride_var = np.var(all_strides)
            if dataset_loaded:
                if (mean_stride < pd_ref['stride_length']) and (stride_var > pd_ref['variability']):
                    gait_status = "Unstable (Parkinson suspected)"
                    insight = "Gait features match Parkinson patient distribution."
                elif (abs(mean_stride - healthy_ref['stride_length']) < 0.05) and (stride_var <= healthy_ref['variability']):
                    gait_status = "Stable (Healthy-like)"
                    insight = "Gait features are consistent with healthy controls."
                else:
                    gait_status = "Borderline"
                    insight = "Gait metrics fall between healthy and Parkinson references."
            else:
                if stride_var < 0.01:
                    gait_status = "Stable"
                    insight = "Your walking pattern is consistent and balanced."
                elif stride_var < 0.05:
                    gait_status = "Slightly Unstable"
                    insight = "Some inconsistencies detected; watch your balance."
                else:
                    gait_status = "Unstable"
                    insight = "High step variability; risk of falling. Consult a specialist."
            st.subheader("1️⃣ Overview / Summary")
            st.markdown(f"**Gait Status:** {gait_status}")
            st.markdown(f"**Interpretation:** {insight}")

            st.subheader("2️⃣ Key Metrics")
            metrics = pd.DataFrame({
                "Metric": ["Average stride length (m)", "Step-to-step variability", "Left-Right Asymmetry (m)"],
                "Value": [f"{mean_stride:.2f}", f"{stride_var:.2f}", f"{asymmetry:.2f}"],
                "Interpretation": [
                    "Average distance traveled by each foot per step.",
                    "Low → steps are consistent. High → irregular steps, risk of imbalance.",
                    "Difference between left & right foot stride lengths; higher indicates asymmetry."
                ]
            })
            st.table(metrics)

            st.subheader("3️⃣ Visual Graphs")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(left_strides, color='blue', alpha=0.7, marker='o', label="Left Foot")
            ax.plot(right_strides, color='orange', alpha=0.7, marker='o', label="Right Foot")
            ax.set_title("Stride Length Over Time")
            ax.set_xlabel("Step Index")
            ax.set_ylabel("Stride Length (m)")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

            if len(left_ankle) > 0 and len(right_ankle) > 0:
                fig2, ax2 = plt.subplots()
                midpoints = (left_ankle + right_ankle) / 2
                ax2.plot(midpoints[:, 0], midpoints[:, 1], color='purple', alpha=0.6)
                ax2.set_title("Stride Path Visualization")
                ax2.set_xlabel("X position (px)")
                ax2.set_ylabel("Y position (px)")
                ax2.invert_yaxis()
                st.pyplot(fig2)

            st.subheader("4️⃣ Insights & Recommendations")
            st.info(insight)
            if gait_status.startswith("Unstable") or asymmetry > 0.05:
                st.warning("⚠️ Consider balance exercises or consult a specialist if gait instability or asymmetry persists.")

        else:
            st.success("✅ No gait keypoints detected. Please upload a clearer video.")
