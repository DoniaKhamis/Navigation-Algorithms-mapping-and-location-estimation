# Navigation, Mapping, and Localization Algorithms 🛰️

![Python](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/Library-OpenCV-5C3EE8?style=flat-square&logo=opencv)
![Ubuntu](https://img.shields.io/badge/OS-Ubuntu-E95420?style=flat-square&logo=ubuntu&logoColor=white)
![Pangolin](https://img.shields.io/badge/Visualization-Pangolin-orange?style=flat-square)

### 🧠 Core Competencies Developed

#### 1. Bayesian Estimation & Filtering
Understanding SLAM as a recursive Bayesian filtering problem.
* **Kalman Filtering:** Implementation and derivation of the **(KF)**, **(EKF)**, and **(UKF)**.
* **Multi-Sensor Fusion:** Fusing Camera and IMU data for robust state estimation.

#### 2. 3D Mapping & Multi-Sensor Calibration
Bridging the gap between raw pixels and spatial awareness.
* **Camera Geometry:** Pinhole camera models, projection matrix mathematics, and lens distortion.
* **Calibration:** Techniques for both **Intrinsic** and **Extrinsic** calibration of Cameras, IMUs, and TOF sensors.
* **Multi-View Geometry:** Applying **Epipolar Geometry** and Structure-from-Motion (SfM) principles.
* **Probabilistic Mapping:** Exploring Sparse vs. Dense 3D representations and Bayesian map estimation.

#### 3. Advanced Probabilistic SLAM
* **Particle Filters:** Monte Carlo Localization (MCL) and importance sampling for non-linear estimation.
* **Graph-Based Methods:** Introduction to **Graph-Based SLAM** and **Pose Graph Optimization (PGO)**.

#### 4. Optimization-Based SLAM & Efficiency
* **Nonlinear Optimization:** **Factor Graphs**, **Bundle Adjustment**, and **Pose Graph Optimization (PGO)** for high-accuracy localization.
* **Computational Scaling:** Investigating sparse representations for large-scale SLAM and information-theoretic approaches to complexity reduction.
