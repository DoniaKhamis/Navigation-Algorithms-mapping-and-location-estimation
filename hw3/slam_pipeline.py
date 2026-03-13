import cv2
import numpy as np
from structures import Frame

class VisualOdometryPipeline:
    def __init__(self, K):
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.detector = cv2.ORB_create(nfeatures=1500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def extract_features(self, frame: Frame):
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        frame.keypoints, frame.descriptors = self.detector.detectAndCompute(gray, None)

    def match_and_filter(self, f1, f2):
        if f1.descriptors is None or f2.descriptors is None: return [], [], []
        matches = self.matcher.knnMatch(f1.descriptors, f2.descriptors, k=2)
        good, p1, p2 = [], [], []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
                p1.append(f1.keypoints[m.queryIdx].pt)
                p2.append(f2.keypoints[m.trainIdx].pt)
        return good, np.array(p1), np.array(p2)

    def calculate_epipolar_error(self, pts1, pts2, F):
        """Calculates geometric distance from points to epipolar lines."""
        if F is None or len(pts1) == 0: return 999.0
        pts1_h = np.column_stack([pts1, np.ones(len(pts1))])
        pts2_h = np.column_stack([pts2, np.ones(len(pts2))])
        lines2 = (F @ pts1_h.T).T 
        error = np.mean(np.abs(np.sum(pts2_h * lines2, axis=1)) / 
                        np.sqrt(lines2[:,0]**2 + lines2[:,1]**2))
        return float(error)
        
    def calculate_reprojection_error(self, pts3d, pts2d, R, t):
        """Requirement 6: Calculates distance between projected 3D points and 2D features."""
        if len(pts3d) == 0: return 999.0
        rvec, _ = cv2.Rodrigues(R)
        projected_pts2d, _ = cv2.projectPoints(pts3d, rvec, t, self.K, None)
        projected_pts2d = projected_pts2d.reshape(-1, 2)
        pts2d = pts2d.reshape(-1, 2)
        error = np.linalg.norm(pts2d - projected_pts2d, axis=1)
        return float(np.mean(error))

    def estimate_pose(self, pts1, pts2):
        """Estimates R, t and returns Epipolar Error before/after RANSAC."""
        F_raw, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
        err_before = self.calculate_epipolar_error(pts1, pts2, F_raw)

        # Tightened threshold to 0.5 to fix the zigzag and force accuracy
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, 
                                       prob=0.999, threshold=0.5)
        
        if E is None or mask is None: 
            return None, None, None, err_before, 999.0
            
        inliers = mask.ravel() == 1
        pts1_in, pts2_in = pts1[inliers], pts2[inliers]
        F_filt = self.K_inv.T @ E @ self.K_inv
        err_after = self.calculate_epipolar_error(pts1_in, pts2_in, F_filt)

        _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, self.K)
        
        # Scale Normalization: Forces consistent movement to stop the zigzag
        if np.linalg.norm(t) > 0:
            t = t / np.linalg.norm(t)
            
        return R, t, inliers, err_before, err_after
    
    def refine_pose(self, R, t, pts3d, pts2d):
        """Requirement 6: Total Update via Gauss-Newton."""
        rvec, _ = cv2.Rodrigues(R)
        rvec_opt, tvec_opt = cv2.solvePnPRefineVVS(pts3d, pts2d, self.K, None, rvec, t)
        R_opt, _ = cv2.Rodrigues(rvec_opt)
        return R_opt, tvec_opt

    def triangulate_points(self, pose1, pose2, pts1, pts2):
        """Requirement 4: SVD-based Triangulation solving AX=0."""
        P1 = self.K @ pose1[:3, :]
        P2 = self.K @ pose2[:3, :]
        pts_3d = []
        for i in range(len(pts1)):
            A = np.array([pts1[i][0]*P1[2,:]-P1[0,:], pts1[i][1]*P1[2,:]-P1[1,:],
                          pts2[i][0]*P2[2,:]-P2[0,:], pts2[i][1]*P2[2,:]-P2[1,:]])
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            if abs(X[3]) > 1e-6:
                X_3d = X[:3]/X[3]
                # Filter points to ensure they are in front of camera
                if 0.5 < X_3d[2] < 50:
                    pts_3d.append(X_3d)
        return np.array(pts_3d)
    
    def relocalize_pnp(self, object_points_3d, image_points_2d):
        """Requirement 5: Perspective-n-Point Relocalization."""
        # Ensure we have the minimum 4 points required by the P3P algorithm
        if len(object_points_3d) < 4: return None, None
        
        success, rvec, tvec, _ = cv2.solvePnPRansac(
            object_points_3d, image_points_2d, self.K, None, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            R, _ = cv2.Rodrigues(rvec)
            return R, tvec
        return None, None
        
    def detect_loop(self, curr_frame, past_frames, min_dist=200): # Increased from 50 to 200
            """Requirement 8: Checks if current frame matches a distant past frame."""
            # We need a long history to prevent matching frames from the same hovering sequence
            if len(past_frames) < min_dist + 10: return None
            
            best_match_count = 0
            best_frame = None
            
            # Search backwards through old frames. 
            # By slicing [:-min_dist], we IGNORE the most recent 200 frames.
            for old_frame in past_frames[:-min_dist][::5]: # Increased step to 5 for speed
                matches, _, _ = self.match_and_filter(old_frame, curr_frame)
                if len(matches) > best_match_count:
                    best_match_count = len(matches)
                    best_frame = old_frame
                    
            # Raised threshold slightly to 35 to ensure it's a confident match
            if best_match_count > 35:
                return best_frame
            return None