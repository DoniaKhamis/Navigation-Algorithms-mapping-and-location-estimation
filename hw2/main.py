import os
import glob
import argparse
import time
import numpy as np
import cv2

# Pangolin import (Linux / WSL compatible)
try:
    import pypangolin as pangolin
except ImportError:
    import pangolin

from OpenGL.GL import *

# =========================
# 1. Frame object
# =========================
class Frame:
    def __init__(self, idx, image):
        self.id = idx
        self.image = image
        self.timestamp = time.time()
        self.keypoints = []
        self.descriptors = None
        self.pose = np.eye(4, dtype=np.float64)
        self.processed = False


# =========================
# Utility functions
# =========================
def load_images(folder):
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, "*" + e)))
    return sorted(paths)


def build_K_from_image(img):
    h, w = img.shape[:2]
    f = 0.9 * max(h, w)
    return np.array([
        [f, 0, w / 2.0],
        [0, f, h / 2.0],
        [0, 0, 1]
    ], dtype=np.float64)


# =========================
# 2. Mono Visual Odometry
# =========================
class MonoVisualOdometry:
    def __init__(self, K):
        self.K = K
        self.T_wc = np.eye(4, dtype=np.float64)
        self.last_t = np.zeros((3, 1))
        self.detector = cv2.ORB_create(nfeatures=3000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def extract(self, frame: Frame):
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        frame.keypoints, frame.descriptors = self.detector.detectAndCompute(gray, None)
        frame.processed = True

    def estimate_motion(self, f1: Frame, f2: Frame):
        if f1.descriptors is None or f2.descriptors is None:
            return None

        matches = self.matcher.knnMatch(f1.descriptors, f2.descriptors, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < 15:
            return None

        pts1 = np.float32([f1.keypoints[m.queryIdx].pt for m in good])
        pts2 = np.float32([f2.keypoints[m.trainIdx].pt for m in good])

        E, _ = cv2.findEssentialMat(
            pts2, pts1, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=0.5
        )
        if E is None:
            return None

        _, R, t, _ = cv2.recoverPose(E, pts2, pts1, self.K)

        # Smooth translation
        alpha = 0.8
        t = alpha * t + (1.0 - alpha) * self.last_t
        self.last_t = t.copy()

        return R, t

    def integrate(self, R, scale):
        forward_cam = np.array([0, 0, 1.0])
        direction_world = R @ forward_cam
        direction_world /= (np.linalg.norm(direction_world) + 1e-6)

        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = direction_world * scale

        self.T_wc = self.T_wc @ np.linalg.inv(T_rel)
        return self.T_wc


# =========================
# 3. Pangolin Viewer
# =========================
class TrajectoryViewer:
    def __init__(self):
        self.poses = []

        pangolin.CreateWindowAndBind(
            "Visual Odometry – Position + Orientation",
            1024, 768
        )
        glEnable(GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(
                1024, 768, 500, 500, 512, 389, 0.1, 1000
            ),
            pangolin.ModelViewLookAt(
                1.5, -1.5, -1.5,
                0, 0, 0,
                0, -1, 0
            )
        )

        self.handler = pangolin.Handler3D(self.scam)
        self.dcam = pangolin.CreateDisplay()

        # ✅ FIXED SetBounds (pypangolin requires Attach)
        self.dcam.SetBounds(
            pangolin.Attach(0.0),
            pangolin.Attach(1.0),
            pangolin.Attach(0.0),
            pangolin.Attach(1.0),
            -1024.0 / 768.0
        )

        self.dcam.SetHandler(self.handler)

    def add_pose(self, T_wc):
        self.poses.append(T_wc.copy())

    def draw_camera_frustum(self, T, size=0.08):
        glLineWidth(1.5)
        glColor3f(0, 0, 1)

        glBegin(GL_LINES)

        v = [T @ np.array([x, y, z, 1]) for x, y, z in [
            (0, 0, 0),
            (size, size, -size * 2),
            (size, -size, -size * 2),
            (-size, -size, -size * 2),
            (-size, size, -size * 2)
        ]]

        for i in range(1, 5):
            glVertex3f(v[0][0], v[0][1], v[0][2])
            glVertex3f(v[i][0], v[i][1], v[i][2])

        for i in range(1, 5):
            glVertex3f(v[i][0], v[i][1], v[i][2])
            glVertex3f(v[1 + (i % 4)][0],
                       v[1 + (i % 4)][1],
                       v[1 + (i % 4)][2])

        glEnd()

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.dcam.Activate(self.scam)

        # Ground grid
        glColor3f(0.5, 0.5, 0.5)
        glLineWidth(1)
        glBegin(GL_LINES)
        for i in np.arange(-5, 5.5, 0.5):
            glVertex3f(i, 0, -5)
            glVertex3f(i, 0, 5)
            glVertex3f(-5, 0, i)
            glVertex3f(5, 0, i)
        glEnd()

        # Trajectory
        if len(self.poses) > 1:
            glColor3f(1, 1, 0)
            glLineWidth(2)
            glBegin(GL_LINE_STRIP)
            for T in self.poses:
                glVertex3f(T[0, 3], T[1, 3], T[2, 3])
            glEnd()

        # Orientation along the path
        for i, T in enumerate(self.poses):
            if i % 5 == 0:
                self.draw_camera_frustum(T)

        pangolin.FinishFrame()

    def should_close(self):
        return pangolin.ShouldQuit()


# =========================
# 4. Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to Dataset_VO")
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--wait", type=int, default=1)
    args = parser.parse_args()

    img_paths = load_images(args.path)
    if not img_paths:
        print("No images found")
        return

    first_img = cv2.imread(img_paths[0])
    K = build_K_from_image(first_img)

    vo = MonoVisualOdometry(K)
    viewer = TrajectoryViewer()

    prev_frame = None
    cv2.namedWindow("Keypoints", cv2.WINDOW_NORMAL)

    for i, p in enumerate(img_paths):
        img = cv2.imread(p)
        if img is None:
            continue

        frame = Frame(i, img)
        vo.extract(frame)

        cv2.imshow(
            "Keypoints",
            cv2.drawKeypoints(img, frame.keypoints, None, color=(0, 255, 0))
        )

        if prev_frame is not None:
            res = vo.estimate_motion(prev_frame, frame)
            if res is not None:
                R, t = res
                frame.pose = vo.integrate(R, args.scale)

        viewer.add_pose(vo.T_wc)
        viewer.draw()

        prev_frame = frame

        if cv2.waitKey(args.wait) & 0xFF == ord('q') or viewer.should_close():
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
