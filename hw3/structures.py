import numpy as np
import cv2
import time
from typing import List, Optional

class Frame:
    """
    Represents a single image frame in the chronological SLAM sequence.
    
    This class encapsulates the 2D image data, the extracted visual features 
    (keypoints and descriptors), and the spatial state of the camera at the 
    exact moment the frame was captured.
    """
    def __init__(self, frame_id: int, image: np.ndarray):
        self.id: int = frame_id
        self.image: np.ndarray = image
        self.timestamp: float = time.time()
        
        # Extracted 2D Features
        self.keypoints: List[cv2.KeyPoint] = []
        self.descriptors: Optional[np.ndarray] = None
        
        # 6-DOF Pose representing the transformation from the World to the Camera.
        # Initialized to identity (no rotation, no translation).
        self.rotation_matrix: np.ndarray = np.eye(3, dtype=np.float64)
        self.translation_vector: np.ndarray = np.zeros((3, 1), dtype=np.float64)
        
        # Status flag indicating if the frame has passed through the pipeline
        self.processed: bool = False

    @property
    def pose(self) -> np.ndarray:
        """
        Dynamically constructs and returns the full 4x4 transformation matrix (T).
        """
        transformation_matrix = np.eye(4, dtype=np.float64)
        transformation_matrix[:3, :3] = self.rotation_matrix
        transformation_matrix[:3, 3:] = self.translation_vector
        return transformation_matrix
class Point:
    """
    Represents a 3D landmark (Map Point) in the global coordinate system.
    
    Generated via triangulation of matched 2D keypoints across multiple frames. 
    It maintains a record of which frames observe it to facilitate bundle adjustment, 
    PnP relocalization, and reprojection error minimization.
    """
    _id_counter = 0  # Class-level counter to ensure unique IDs

    def __init__(self, point_3d: np.ndarray):
        Point._id_counter += 1
        self.id: int = Point._id_counter
        
        # The 3D coordinates (X, Y, Z) in the global frame
        self.point: np.ndarray = np.array(point_3d, dtype=np.float64).reshape(3)
        
        # List of Frame objects that have observed this 3D point
        self.frames: List['Frame'] = []

    def add_observation(self, frame: 'Frame') -> None:
        """
        Registers a frame that observes this 3D map point.
        """
        if frame not in self.frames:
            self.frames.append(frame)
            
class Map:
    """
    The global map containing all tracked 3D landmarks and processed camera frames.
    
    Serves as the central state of the SLAM system, enabling PnP relocalization 
    by querying historical points, and allowing global map visualization.
    """
    def __init__(self):
        self.points: List[Point] = []  
        self.frames: List[Frame] = []  

    def add_frame(self, frame: Frame) -> None:
        """Adds a processed frame to the global map."""
        self.frames.append(frame)

    def add_point(self, point: Point) -> None:
        """Adds a newly triangulated 3D point to the global map."""
        self.points.append(point)
        
    def get_all_3d_points(self) -> np.ndarray:
        """
        Helper method to extract all X, Y, Z coordinates for Pangolin rendering.
        Returns an Nx3 numpy array.
        """
        if not self.points:
            return np.empty((0, 3))
        return np.array([p.point for p in self.points])