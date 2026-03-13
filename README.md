# Navigation, Mapping, and Localization Algorithms 🛰️

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
