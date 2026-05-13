import cv2
import numpy as np
from typing import Optional
from loguru import logger
from .calibration_system import SmartDepthCalibrator
from .depth_estimator import DepthEstimator

class EasyCalibrator:
    """Easy-to-use calibration interface"""
    
    def __init__(self, depth_estimator: DepthEstimator):
        self.depth_estimator = depth_estimator
        self.calibrator = SmartDepthCalibrator()
        
        # Common objects with known typical sizes for automatic calibration
        self.known_objects = {
            'bottle': 0.25,  # 25cm typical height
            'person': 1.7,   # 1.7m typical height
            'chair': 0.8,    # 80cm typical height
            'cup': 0.1,      # 10cm typical height
            'book': 0.02,    # 2cm typical thickness
            'laptop': 0.02,  # 2cm typical thickness when closed
        }
    
    def quick_calibrate_with_known_distances(self, measurements: list):
        """
        Quick calibration with known measurements
        
        Args:
            measurements: List of tuples [(frame, bbox, real_distance, confidence), ...]
        """
        for frame, bbox, real_distance, confidence in measurements:
            depth_map = self.depth_estimator.estimate_depth(frame)
            depth_value = self._get_depth_from_bbox(depth_map, bbox)
            
            if depth_value > 0:
                self.calibrator.add_calibration_point(
                    depth_value, real_distance, confidence
                )
        
        success = self.calibrator.auto_calibrate()
        return success
    
    def smart_object_calibration(self, frame: np.ndarray, detected_objects: list, 
                               user_distance: Optional[float] = None):
        """
        Smart calibration using object recognition and typical sizes
        
        Args:
            frame: Current frame
            detected_objects: List of detected objects
            user_distance: Optional user-provided distance for current closest object
        """
        depth_map = self.depth_estimator.estimate_depth(frame)
        
        if user_distance:
            # Use user-provided distance for closest object
            if detected_objects:
                closest_obj = min(detected_objects, key=lambda obj: 
                                obj.distance if obj.distance else float('inf'))
                
                depth_value = self._get_depth_from_bbox(depth_map, closest_obj.bbox)
                if depth_value > 0:
                    self.calibrator.add_calibration_point(
                        depth_value, user_distance, 0.9, closest_obj.class_name
                    )
                    logger.info(f"Calibration point added: {closest_obj.class_name} at {user_distance}m")
        
        # Auto-calibrate using known object sizes
        for obj in detected_objects:
            if obj.class_name in self.known_objects:
                depth_value = self._get_depth_from_bbox(depth_map, obj.bbox)
                estimated_distance = self._estimate_distance_from_size(obj, frame.shape)
                
                if depth_value > 0 and estimated_distance > 0:
                    # Lower confidence for automatic estimates
                    self.calibrator.add_calibration_point(
                        depth_value, estimated_distance, 0.6, obj.class_name
                    )
    
    def _get_depth_from_bbox(self, depth_map: np.ndarray, bbox) -> float:
        """Extract depth value from bounding box region"""
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0
        
        roi_depth = depth_map[y1:y2, x1:x2]
        
        # Use central region
        center_h, center_w = roi_depth.shape
        if center_h > 10 and center_w > 10:
            margin_h, margin_w = center_h // 4, center_w // 4
            center_roi = roi_depth[margin_h:center_h-margin_h, margin_w:center_w-margin_w]
            return float(np.median(center_roi))
        else:
            return float(np.median(roi_depth))
    
    def _estimate_distance_from_size(self, obj, frame_shape) -> float:
        """Estimate distance based on object size in image vs known real size"""
        try:
            bbox_height = obj.bbox - obj.bbox
            frame_height = frame_shape
            
            # Simple pinhole camera model estimation
            # This is a rough approximation
            if obj.class_name in self.known_objects:
                real_height = self.known_objects[obj.class_name]
                
                # Approximate focal length (calibrate this for your camera)
                focal_length_pixels = 500  # Typical value, adjust as needed
                
                estimated_distance = (real_height * focal_length_pixels) / bbox_height
                
                # Clamp to reasonable range
                return max(0.5, min(estimated_distance, 10.0))
        
        except Exception as e:
            logger.warning(f"Size-based distance estimation failed: {e}")
        
        return 0
    
    def get_calibration_status(self) -> dict:
        """Get current calibration status"""
        quality = self.calibrator.get_calibration_quality()
        return {
            'is_calibrated': self.calibrator.active_model is not None,
            'quality_score': quality['quality_score'],
            'sample_count': quality['sample_count'],
            'model_type': quality.get('model_type', 'none'),
            'accuracy': quality.get('accuracy_rmse', 0),
            'recommendations': self._get_calibration_recommendations(quality)
        }
    
    def _get_calibration_recommendations(self, quality: dict) -> list:
        """Get recommendations to improve calibration"""
        recommendations = []
        
        if quality['sample_count'] < 10:
            recommendations.append("Add more calibration points for better accuracy")
        
        if quality['coverage_score'] < 0.5:
            recommendations.append("Calibrate with objects at different distances")
        
        if quality['quality_score'] < 0.7:
            recommendations.append("Current calibration may be inaccurate, consider recalibrating")
        
        return recommendations
