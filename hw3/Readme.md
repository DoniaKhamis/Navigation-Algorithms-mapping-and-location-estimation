# HW3: Monocular RGB-SLAM and 3D Reconstruction 🛰️

This project implements a complete **Simultaneous Localization and Mapping (SLAM)** system using a monocular RGB camera sequence. The system performs trajectory estimation, 3D point cloud mapping, and global optimization to ensure a consistent environmental reconstruction.

## 📋 Project Scope
The objective is to build a full SLAM pipeline that handles real-time localization and environmental mapping. Unlike pure Visual Odometry, this system maintains a global map and corrects for drift over time.

### Core Requirements:
* **Feature Matching:** Extracting and matching features across sequential frames with outlier filtering.
* **3D Reconstruction:** Using **Triangulation** to build a global **Point Cloud** map.
* **Localization (PnP):** Periodically re-localizing the drone using Perspective-n-Point (PnP) algorithms.
* **Loop Closure:** Identifying previously visited locations to correct cumulative trajectory drift.

---

## ⚙️ Mathematical & Optimization Framework

### 1. Geometric Errors
The system tracks two critical error metrics to ensure accuracy:
* **Epipolar Error:** Calculated for descriptor matches to validate the geometric consistency between frames.
* **Reprojection Error:** Monitored before and after optimization to measure the quality of the 3D-to-2D projection.

### 2. Nonlinear Optimization
To refine the map and trajectory, the system utilizes the **Gradient Descent**

---

## 🏗️ Software Architecture

### Data Structures
The implementation uses a hierarchical object-oriented approach:
* **`FRAME` Class:** Stores keypoints, descriptors, and the 6DOF camera pose ($4\times4$ Eigen Matrix).
* **`Point` Class:** Represents a unique 3D point in the global coordinate system, tracking which frames it appears in.
* **`Map` Class:** A container for the entire Point Cloud and the collection of processed frames.

### Coordinate System
* **Z-axis:** Forward/Backward
* **X-axis:** Right/Left
* **Y-axis:** Up/Down

---

## 🖥️ Visualizations
The system provides a real-time dual-window display using **Pangolin**:
1.  **Feature Viewer:** Live camera feed showing tracked keypoints and matching status.
2.  **SLAM Map:** A 3D visualization showing the drone's cumulative trajectory and the evolving point cloud map.
