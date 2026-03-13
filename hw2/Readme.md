# HW2: Epipolar Visual Odometry for Drone Path Estimation 🛸

This project implements a **Visual Odometry (VO)** system to estimate a drone's 3D trajectory from a sequence of images. By analyzing the geometric relationship between consecutive frames using **Epipolar Geometry**, the system reconstructs the camera's motion in real-time.

## 📋 Project Overview
The system processes a sequential image dataset (`Dataset_VO`) where each image represents a small displacement from the previous frame. The goal is to calculate the relative motion and visualize the resulting path without building a full environmental map.

### Key Features:
* **Feature Detection & Matching:** Identifying and tracking keypoints (ORB, SIFT, etc.) across frames.
* **Epipolar Geometry:** Computing relative motion (Rotation and Translation) between image pairs.
* **Real-time Visualization:** Using the **Pangolin** library to display the drone's trajectory as it processes the image sequence.

---

## 🏗️ System Architecture

### 1. The `FRAME` Class
The implementation is centered around a robust `FRAME` class that represents a single point in the image sequence. Each frame object stores:
* **KeyPoints:** Spatial features found within the image.
* **Descriptors:** Feature vectors used for cross-frame matching.
* **ID & Timestamp:** Unique identifiers for chronological processing.
* **Pose:** The 6DOF camera state (Rotation Matrix $3\times3$ and Translation Vector $3\times1$).

### 2. Coordinate System
The system uses a flexible 3D coordinate frame, with the following preferred orientation:
* **Z-axis:** Forward / Backward
* **X-axis:** Right / Left
* **Y-axis:** Up / Down

### 3. Camera Matrix
To simplify the pipeline, the **Intrinsic Matrix** is derived from the image dimensions rather than a manual calibration process.

---

## 🖥️ Visual Outputs
Upon execution, the system launches two synchronized windows:
1. **Feature Tracker:** Displays the current camera feed with highlighted keypoints being tracked in real-time.
2. **Trajectory Map:** A 3D environment (powered by Pangolin) that visualizes the estimated path and current camera pose.
