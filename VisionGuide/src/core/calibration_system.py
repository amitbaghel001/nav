import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict
import json
import time
from pathlib import Path
from loguru import logger

@dataclass
class CalibrationPoint:
    """Single calibration data point"""
    depth_value: float
    real_distance_meters: float
    confidence_score: float
    timestamp: float
    frame_size: Tuple[int, int]
    object_class: str

@dataclass
class CalibrationModel:
    """Calibration model parameters"""
    model_type: str  # 'inverse', 'power', 'polynomial'
    parameters: List[float]
    accuracy_score: float
    calibration_date: str
    sample_count: int

class SmartDepthCalibrator:
    """Intelligent calibration system for depth-to-distance conversion"""
    
    def __init__(self, save_path: str = "calibration_data.json"):
        self.calibration_points: List[CalibrationPoint] = []
        self.models: Dict[str, CalibrationModel] = {}
        self.active_model: Optional[CalibrationModel] = None
        self.save_path = Path(save_path)
        
        # Load existing calibration if available
        self.load_calibration()
        
        # Calibration parameters
        self.min_points_required = 5
        self.max_points_stored = 100
        self.confidence_threshold = 0.7
        
        logger.info("Smart depth calibrator initialized")
    
    def add_calibration_point(self, depth_value: float, real_distance: float, 
                            confidence: float, object_class: str = "unknown",
                            frame_size: Tuple[int, int] = (640, 480)) -> bool:
        """Add a new calibration point with validation"""
        
        # Validate inputs
        if depth_value <= 0 or real_distance <= 0:
            logger.warning("Invalid calibration values provided")
            return False
        
        if confidence < self.confidence_threshold:
            logger.warning(f"Low confidence calibration point rejected: {confidence}")
            return False
        
        # Create calibration point
        point = CalibrationPoint(
            depth_value=depth_value,
            real_distance_meters=real_distance,
            confidence_score=confidence,
            timestamp=time.time(),
            frame_size=frame_size,
            object_class=object_class
        )
        
        # Add to collection (with size limit)
        self.calibration_points.append(point)
        if len(self.calibration_points) > self.max_points_stored:
            # Remove oldest low-confidence points first
            self.calibration_points.sort(key=lambda x: (x.confidence_score, x.timestamp))
            self.calibration_points = self.calibration_points[-self.max_points_stored:]
        
        logger.info(f"Calibration point added: depth={depth_value:.3f}, distance={real_distance:.2f}m")
        
        # Auto-recalibrate if we have enough points
        if len(self.calibration_points) >= self.min_points_required:
            self.auto_calibrate()
        
        # Save updated data
        self.save_calibration()
        return True
    
    def auto_calibrate(self) -> bool:
        """Automatically determine best calibration model"""
        if len(self.calibration_points) < self.min_points_required:
            logger.warning("Insufficient calibration points for auto-calibration")
            return False
        
        # Extract data arrays
        depths = np.array([p.depth_value for p in self.calibration_points])
        distances = np.array([p.real_distance_meters for p in self.calibration_points])
        weights = np.array([p.confidence_score for p in self.calibration_points])
        
        # Test different models and choose the best one
        models_to_test = [
            ('inverse', self._fit_inverse_model),
            ('power', self._fit_power_model),
            ('polynomial', self._fit_polynomial_model)
        ]
        
        best_model = None
        best_score = float('inf')
        
        for model_name, fit_function in models_to_test:
            try:
                params, score = fit_function(depths, distances, weights)
                logger.info(f"{model_name} model score: {score:.4f}")
                
                if score < best_score:
                    best_score = score
                    best_model = CalibrationModel(
                        model_type=model_name,
                        parameters=params.tolist(),
                        accuracy_score=score,
                        calibration_date=time.strftime("%Y-%m-%d %H:%M:%S"),
                        sample_count=len(self.calibration_points)
                    )
            except Exception as e:
                logger.warning(f"Failed to fit {model_name} model: {e}")
        
        if best_model:
            self.active_model = best_model
            self.models[best_model.model_type] = best_model
            logger.info(f"Best model selected: {best_model.model_type} with score {best_score:.4f}")
            return True
        
        return False
    
    def _fit_inverse_model(self, depths: np.ndarray, distances: np.ndarray, 
                          weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fit inverse model: distance = a / (depth + b) + c"""
        def model_func(depth, a, b, c):
            return a / (depth + b) + c
        
        try:
            from scipy.optimize import curve_fit
            popt, _ = curve_fit(model_func, depths, distances, 
                              sigma=1/weights, maxfev=5000,
                              bounds=([0, 0, 0], [100, 10, 10]))
            
            # Calculate R² score
            predicted = model_func(depths, *popt)
            score = np.sqrt(np.mean((distances - predicted)**2))
            
            return popt, score
        except ImportError:
            # Fallback without scipy
            logger.warning("Scipy not available, using simple inverse model")
            # Simple least squares for a/x model
            a = np.sum(weights * depths * distances) / np.sum(weights * depths**2)
            score = np.sqrt(np.mean((distances - a/depths)**2))
            return np.array([a, 0.1, 0]), score
    
    def _fit_power_model(self, depths: np.ndarray, distances: np.ndarray, 
                        weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fit power model: distance = a * depth^b + c"""
        def model_func(depth, a, b, c):
            return a * (depth ** b) + c
        
        try:
            from scipy.optimize import curve_fit
            popt, _ = curve_fit(model_func, depths, distances, 
                              sigma=1/weights, maxfev=5000,
                              bounds=([-100, -5, -10], [100, 5, 10]))
            
            predicted = model_func(depths, *popt)
            score = np.sqrt(np.mean((distances - predicted)**2))
            
            return popt, score
        except:
            # Fallback linear relationship
            coeffs = np.polyfit(depths, distances, 1, w=weights)
            score = np.sqrt(np.mean((distances - np.polyval(coeffs, depths))**2))
            return np.array([coeffs[0], 1, coeffs[1]]), score
    
    def _fit_polynomial_model(self, depths: np.ndarray, distances: np.ndarray, 
                             weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fit polynomial model: distance = a*depth² + b*depth + c"""
        coeffs = np.polyfit(depths, distances, 2, w=weights)
        predicted = np.polyval(coeffs, depths)
        score = np.sqrt(np.mean((distances - predicted)**2))
        
        return coeffs, score
    
    def depth_to_distance(self, depth_value: float) -> float:
        """Convert depth value to real distance using active model"""
        if not self.active_model:
            # No calibration, use default
            return self._default_conversion(depth_value)
        
        try:
            params = np.array(self.active_model.parameters)
            
            if self.active_model.model_type == 'inverse':
                a, b, c = params
                distance = a / (depth_value + b) + c
                
            elif self.active_model.model_type == 'power':
                a, b, c = params
                distance = a * (depth_value ** b) + c
                
            elif self.active_model.model_type == 'polynomial':
                distance = np.polyval(params, depth_value)
                
            else:
                distance = self._default_conversion(depth_value)
            
            # Clamp to reasonable range
            return max(0.3, min(distance, 20.0))
            
        except Exception as e:
            logger.error(f"Calibration conversion failed: {e}")
            return self._default_conversion(depth_value)
    
    def _default_conversion(self, depth_value: float) -> float:
        """Default conversion when no calibration is available"""
        base_distance = 3.0
        return base_distance / (depth_value + 0.1)
    
    def get_calibration_quality(self) -> Dict[str, float]:
        """Get metrics about calibration quality"""
        if not self.active_model:
            return {
                'quality_score': 0.0,
                'sample_count': len(self.calibration_points),
                'coverage_score': 0.0
            }
        
        # Calculate coverage (how well we cover the distance range)
        distances = [p.real_distance_meters for p in self.calibration_points]
        if distances:
            distance_range = max(distances) - min(distances)
            coverage_score = min(1.0, distance_range / 10.0)  # Normalize to 10m range
        else:
            coverage_score = 0.0
        
        # Overall quality score
        accuracy_factor = 1.0 - min(1.0, self.active_model.accuracy_score / 2.0)
        sample_factor = min(1.0, len(self.calibration_points) / 20.0)
        quality_score = (accuracy_factor * 0.5 + sample_factor * 0.3 + coverage_score * 0.2)
        
        return {
            'quality_score': quality_score,
            'sample_count': len(self.calibration_points),
            'coverage_score': coverage_score,
            'accuracy_rmse': self.active_model.accuracy_score if self.active_model else 0.0,
            'model_type': self.active_model.model_type if self.active_model else 'none'
        }
    
    def save_calibration(self):
        """Save calibration data to file"""
        try:
            data = {
                'calibration_points': [asdict(p) for p in self.calibration_points],
                'models': {name: asdict(model) for name, model in self.models.items()},
                'active_model': asdict(self.active_model) if self.active_model else None
            }
            
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Calibration data saved to {self.save_path}")
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
    
    def load_calibration(self):
        """Load calibration data from file"""
        try:
            if not self.save_path.exists():
                logger.info("No existing calibration data found")
                return
            
            with open(self.save_path, 'r') as f:
                data = json.load(f)
            
            # Load calibration points
            self.calibration_points = [
                CalibrationPoint(**point) for point in data.get('calibration_points', [])
            ]
            
            # Load models
            self.models = {
                name: CalibrationModel(**model_data) 
                for name, model_data in data.get('models', {}).items()
            }
            
            # Load active model
            if data.get('active_model'):
                self.active_model = CalibrationModel(**data['active_model'])
            
            logger.info(f"Loaded {len(self.calibration_points)} calibration points and {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
    
    def reset_calibration(self):
        """Reset all calibration data"""
        self.calibration_points.clear()
        self.models.clear()
        self.active_model = None
        
        try:
            if self.save_path.exists():
                self.save_path.unlink()
        except Exception as e:
            logger.error(f"Failed to delete calibration file: {e}")
        
        logger.info("Calibration data reset")
