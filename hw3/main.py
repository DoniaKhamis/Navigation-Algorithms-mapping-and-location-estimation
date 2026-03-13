import cv2
import numpy as np
import os
import sys
import pypangolin as pangolin
from OpenGL.GL import *
from structures import Frame, Point, Map
from slam_pipeline import VisualOdometryPipeline

class Viewer3D:
    def __init__(self):
        # Initialize Pangolin window for real-time visualization (Req 91-92)
        pangolin.CreateWindowAndBind("SLAM 3D Map", 1024, 768)
        glEnable(GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(1024, 768, 420, 420, 512, 384, 0.1, 1000),
            pangolin.ModelViewLookAt(0, -20, -60, 0, 0, 0, 0, -1, 0)
        )
        self.handler = pangolin.Handler3D(self.scam)
        self.view = pangolin.CreateDisplay().SetBounds(
            pangolin.Attach(0.0), pangolin.Attach(1.0), 
            pangolin.Attach(0.0), pangolin.Attach(1.0), -1024.0/768.0
        ).SetHandler(self.handler)

    def draw(self, slam_map: Map):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.0, 1.0) 
        self.view.Activate(self.scam)

        # --- Requirement 4: 3D Visualization (Camera Head) ---
        # Draws only the current camera pose as a yellow frustum
        if len(slam_map.frames) > 0:
            current_f = slam_map.frames[-1] 
            glPushMatrix()
            glMultMatrixd(np.linalg.inv(current_f.pose).T)
            w, h, z = 0.5, 0.4, 0.6
            glColor3f(1.0, 1.0, 0.0) 
            glLineWidth(2)
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0); glVertex3f(w, h, z)
            glVertex3f(0, 0, 0); glVertex3f(w, -h, z)
            glVertex3f(0, 0, 0); glVertex3f(-w, -h, z)
            glVertex3f(0, 0, 0); glVertex3f(-w, h, z)
            glVertex3f(w, h, z); glVertex3f(w, -h, z)
            glVertex3f(w, -h, z); glVertex3f(-w, -h, z)
            glVertex3f(-w, -h, z); glVertex3f(-w, h, z)
            glVertex3f(-w, h, z); glVertex3f(w, h, z)
            glEnd()
            glPopMatrix()

        # --- Requirement 3: Cumulative Trajectory ---
        if len(slam_map.frames) > 1:
            glLineWidth(2)
            glColor3f(0.0, 1.0, 0.0)
            glBegin(GL_LINE_STRIP)
            for f in slam_map.frames:
                p_inv = np.linalg.inv(f.pose)
                glVertex3f(p_inv[0, 3], p_inv[1, 3], p_inv[2, 3])
            glEnd()

        # --- Requirement 4: 3D Point Cloud ---
        pts = slam_map.get_all_3d_points()
        if len(pts) > 0:
            glPointSize(2)
            glColor3f(1.0, 1.0, 1.0)
            glBegin(GL_POINTS)
            for p in pts:
                glVertex3f(p[0], p[1], p[2])
            glEnd()

        pangolin.FinishFrame()

def main():
    # Load dataset images (Req 34-35)
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "Dataset_VO"
    image_sequence = []
    rgb_txt = os.path.join(dataset_path, "rgb.txt")
    if os.path.exists(rgb_txt):
        with open(rgb_txt, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    parts = line.split()
                    image_sequence.append((parts[0], os.path.join(dataset_path, parts[1])))

    if not image_sequence: return print("No images found.")

    # Requirement 5: Internal Matrix (K) setup
    first_img = cv2.imread(image_sequence[0][1])
    K = np.array([[800, 0, first_img.shape[1]/2], [0, 800, first_img.shape[0]/2], [0, 0, 1]])
    
    pipeline = VisualOdometryPipeline(K)
    slam_map = Map()
    viewer = Viewer3D()
    prev_frame = None
    last_loop_closure_frame = -999 
    
    for i, (ts, img_path) in enumerate(image_sequence):
        img = cv2.imread(img_path)
        if img is None: continue

        curr_frame = Frame(frame_id=i, image=img)
        
        # --- Requirement 1: Feature Extraction ---
        pipeline.extract_features(curr_frame)

        if prev_frame is not None:
            # --- Requirement 1: Feature Matching ---
            matches, pts1, pts2 = pipeline.match_and_filter(prev_frame, curr_frame)
            
            if len(matches) > 15:
                # --- Requirement 2: Epipolar Error Calculation ---
                res = pipeline.estimate_pose(pts1, pts2)
                if res[0] is not None:
                    R, t, inliers, err_b, err_a = res
                    print(f"Frame {i}: Epipolar Error | Before: {err_b:.4f} -> After RANSAC: {err_a:.4f}")

                    # --- Requirement 3: Trajectory (Pose Update) ---
                    T_rel = np.eye(4)
                    T_rel[:3,:3], T_rel[:3,3:] = R, t
                    new_pose = T_rel @ prev_frame.pose
                    curr_frame.rotation_matrix = new_pose[:3,:3]
                    curr_frame.translation_vector = new_pose[:3, 3:]
                    
                    # --- Requirement 4: 3D Reconstruction (Triangulation) ---
                    dist_moved = np.linalg.norm(curr_frame.translation_vector - prev_frame.translation_vector)
                    if dist_moved > 0.1: 
                        pts1_in, pts2_in = pts1[inliers], pts2[inliers]
                        new_pts = pipeline.triangulate_points(prev_frame.pose, curr_frame.pose, pts1_in, pts2_in)
                        for p in new_pts:
                            slam_map.add_point(Point(p))

                    # --- Requirement 5 & 6 & 7: PnP, Reprojection Error, and Optimization ---
                    if i % 10 == 0 and len(slam_map.points) > 50:
                        num_matches = len(pts2_in)
                        if num_matches >= 8:
                            # Requirement 5: Re-localization via PnP
                            world_pts = np.array([p.point for p in slam_map.points[-num_matches:]], dtype=np.float32).reshape(-1, 1, 3)
                            img_pts = np.array(pts2_in, dtype=np.float32).reshape(-1, 1, 2)

                            if world_pts.shape[0] == img_pts.shape[0]:
                                R_pnp, t_pnp = pipeline.relocalize_pnp(world_pts, img_pts)
                                
                                if R_pnp is not None:
                                    # Requirement 6: Reprojection Error Before
                                    err_rep_before = pipeline.calculate_reprojection_error(world_pts, img_pts, R_pnp, t_pnp)
                                    
                                    # Requirement 7: Optimization (Gauss-Newton)
                                    R_opt, t_opt = pipeline.refine_pose(R_pnp, t_pnp, world_pts, img_pts)
                                    
                                    # Requirement 6: Reprojection Error After
                                    err_rep_after = pipeline.calculate_reprojection_error(world_pts, img_pts, R_opt, t_opt)
                                    
                                    # Check drift to ensure optimization is stable
                                    drift = np.linalg.norm(t_opt - curr_frame.translation_vector)
                                    if drift < 2.0: 
                                        curr_frame.rotation_matrix = R_opt
                                        curr_frame.translation_vector = t_opt
                                        print(f"--- Frame {i}: Optimization Applied | Reprojection Error {err_rep_before:.4f} -> {err_rep_after:.4f} ---")

            # --- Requirement 8: Loop Closure Detection ---
            if i % 5 == 0 and (i - last_loop_closure_frame) > 50:
                loop_frame = pipeline.detect_loop(curr_frame, slam_map.frames)
                if loop_frame:
                    print(f"\n!!! LOOP CLOSURE DETECTED: Frame {i} matches Frame {loop_frame.id} !!!\n")
                    curr_frame.rotation_matrix = loop_frame.rotation_matrix
                    curr_frame.translation_vector = loop_frame.translation_vector
                    last_loop_closure_frame = i

        slam_map.add_frame(curr_frame)
        viewer.draw(slam_map)
        
        # Requirement 1 & 6: Display Keypoints and Window 1 (Tracking)
        cv2.imshow("Tracking", cv2.drawKeypoints(img, curr_frame.keypoints, None, color=(0,255,0)))
        
        if cv2.waitKey(1) == 27 or pangolin.ShouldQuit(): break
        prev_frame = curr_frame

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()