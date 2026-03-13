# HW1: Drone Path Estimation using Kalman Filter 🚁

This project implements a **Kalman Filter (KF)** and **Extended Kalman Filter (EKF)** to reconstruct the real-time flight path of a Tello drone. By fusing inertial data with noisy measurements, the system provides a smoothed estimation of the drone's trajectory in a 3D environment.

## 📋 Task Overview
The system processes two JSON data streams from a drone autonomous navigation system:
1. **Inertial Data:** Clean data containing velocity, acceleration, and Euler angles.
2. **Noisy Measurements:** The same data format but containing significant noise and randomness.

The filter performs a recursive **Prediction** and **Update** cycle to estimate the drone's 6-Degree of Freedom (6DOF) state.

---

## ⚙️ System Configuration

### 1. State Vector
The system state is defined as a 6-component vector:
$$x = [x, y, z, \text{pitch}, \text{roll}, \text{yaw}]^T$$

### 2. Coordinate Systems 🌐
The implementation handles the transformation between the fixed World frame (aligned with the drone's starting position) and the IMU frame:

| Axis | World/Drone System | IMU System |
| :--- | :--- | :--- |
| **X** | Lateral Right (+) | Backward (+) |
| **Y** | Vertical Down (+) | Lateral Left (+) |
| **Z** | Forward (+) | Vertical Top (+) |

* **Note:** Yaw is positive in the clockwise direction.
* **Note:** Velocity is treated as negative when the IMU axes are inverted relative to the World system.

### 3. Filter Parameters
* **Process Noise ($Q$):** Assumed to be very low ($0.00001$).
* **Measurement Noise ($R_k$):** The filter is tested across varying noise levels to analyze stability: $R_k \in \{0.05, 5, 50, 500\}$.

---

## 📊 Outputs & Analysis
The implementation generates a PDF report featuring:
* **Trajectory Visualization:** A comparison between the **Predicted path (Blue)** and the **Updated path (Red)**.
* **Kalman Gain:** A graph depicting the Gain value at each step to show how the filter balances prediction vs. measurement.
* **Sensitivity Analysis:** Observations on how increasing measurement noise ($R_k$) affects the smoothing of the trajectory.
