import cv2
import numpy as np
from typing import Optional
import threading
import time
from loguru import logger
import sys
from pathlib import Path
import signal

from core.easy_calibrator import EasyCalibrator

# Add src to path
sys.path.append(str(Path(__file__).parent))

from core.object_detector import ObjectDetector, PersonalizedObjectDetector
from core.depth_estimator import DepthEstimator
from core.scene_analyzer import SceneAnalyzer, format_for_audio
from core.scene_tracker import SceneTracker
from core.audio_processor import AudioProcessor, AudioPriority
from utils.config import config

from ui.main_window import VisionGuideMainWindow
from PyQt5.QtWidgets import QApplication
import sys

class VisionGuideAI:
    def __init__(self):
        """Initialize NAVIS AI system"""
        self.running = False
        self.camera = None
        self.should_exit = False
        self.last_description = ""
        
        # Initialize components
        self.object_detector = ObjectDetector(config.model.yolo_model_path)
        self.depth_estimator = DepthEstimator(model_type=config.model.depth_model_type)
        self.scene_analyzer = SceneAnalyzer()
        self.scene_tracker = SceneTracker()  # NEW: Smart scene tracker
        self.audio_processor = AudioProcessor()
        self.personalized_detector = PersonalizedObjectDetector()
        
        # Threading
        self.detection_thread = None
        
        # Smart announcement settings
        self.auto_announce_changes = True
        self.periodic_summary_interval = 30.0  # Summary every 30 seconds
        self.last_summary_time = 0

        self.easy_calibrator = EasyCalibrator(self.depth_estimator)

        # Add frame sharing
        self.current_frame = None
        self.processed_frame = None
        self.frame_lock = threading.Lock()
        
        logger.info("NAVIS AI initialized successfully")
    
    def start_camera(self) -> bool:
        """Start camera capture"""
        try:
            self.camera = cv2.VideoCapture(config.camera_index)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, config.fps)
            
            if not self.camera.isOpened():
                logger.error("Failed to open camera")
                return False
            
            logger.info("Camera started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        if self.camera:
            self.camera.release()
            self.camera = None
            logger.info("Camera stopped")
    
    def process_frame(self, frame: np.ndarray) -> str:
        """Process single frame and return audio description"""
        try:
            # Detect objects
            detected_objects = self.object_detector.detect_objects(
                frame, 
                config.model.confidence_threshold,
                config.model.iou_threshold
            )
            
            # Estimate depth
            depth_map = self.depth_estimator.estimate_depth(frame)
            
            # Calculate distances for detected objects
            for obj in detected_objects:
                obj.distance = self.depth_estimator.get_object_distance(
                    depth_map, obj.bbox, config.navigation.step_size
                )
            
            # Recognize known faces
            recognized_faces = self.personalized_detector.recognize_faces(frame)
            
            # Analyze scene
            scene_context = self.scene_analyzer.analyze_scene(detected_objects, depth_map)
            
            # Format for audio
            audio_description = format_for_audio(scene_context, detected_objects)
            
            # Add personalized recognitions
            if recognized_faces:
                face_descriptions = []
                for face in recognized_faces:
                    if face['name'] != "Unknown":
                        face_descriptions.append(f"{face['name']} is present")
                if face_descriptions:
                    audio_description = ". ".join(face_descriptions) + ". " + audio_description
            
            return audio_description
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return "Unable to process scene"
    
    def run_detection_loop(self):
        """Modified detection loop for GUI with periodic summaries"""
        while self.running and not self.should_exit:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue

                # Fix camera mirroring
                frame = cv2.flip(frame, 1)

                # Store current frame for GUI
                with self.frame_lock:
                    self.current_frame = frame.copy()

                # Detect objects
                detected_objects = self.object_detector.detect_objects(
                    frame,
                    config.model.confidence_threshold,
                    config.model.iou_threshold
                )

                # Estimate depth for detected objects
                depth_map = self.depth_estimator.estimate_depth(frame)
                for obj in detected_objects:
                    obj.distance = self.depth_estimator.get_object_distance(
                        depth_map, obj.bbox, config.navigation.step_size
                    )

                # Create processed frame with detections
                display_frame = self.object_detector.draw_detections(
                    frame.copy(), detected_objects
                )

                # Add distance information to display
                for obj in detected_objects:
                    if obj.distance:
                        x1, y1, x2, y2 = obj.bbox
                        distance_steps = obj.distance
                        if distance_steps < 1.5:
                            distance_text = f"very close"
                            color = (0, 0, 255)
                        elif distance_steps < 3.0:
                            distance_text = f"{distance_steps:.1f} steps"
                            color = (0, 165, 255)
                        else:
                            distance_text = f"{distance_steps:.0f} steps"
                            color = (0, 255, 0)
                        cv2.putText(display_frame, distance_text,
                                (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Store processed frame for GUI
                with self.frame_lock:
                    self.processed_frame = display_frame.copy()

                # Update scene tracker and handle announcements
                changes = self.scene_tracker.update_scene(detected_objects)
                if changes and self.auto_announce_changes:
                    if self.scene_tracker.should_announce():
                        meaningful_changes = []
                        for change_type, message in changes.items():
                            if message and not self._is_rapid_change(change_type):
                                meaningful_changes.append(message)
                        if meaningful_changes:
                            announcement = ". ".join(meaningful_changes[:2])
                            self.audio_processor.speak_async(announcement, AudioPriority.HIGH)
                            self.last_description = announcement
                            logger.info(f"Smart announcement: {announcement}")

                # ADD THIS: Periodic scene summary (RESTORED)
                current_time = time.time()
                if current_time - self.last_summary_time >= self.periodic_summary_interval:
                    summary = self.scene_tracker.get_scene_summary()
                    if summary and "analyzing" not in summary.lower():
                        # Only announce if not currently speaking or listening
                        if (not self.audio_processor.is_speaking and 
                            not getattr(self.audio_processor, '_is_listening', False)):
                            self.audio_processor.speak_async(f"Scene update: {summary}", AudioPriority.LOW)
                            logger.info(f"Periodic scene summary: {summary}")
                    self.last_summary_time = current_time

                time.sleep(0.03)

            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                time.sleep(0.1)


    def get_current_frame(self):
        """Get current raw frame"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_processed_frame(self):
        """Get processed frame with detections"""
        with self.frame_lock:
            return self.processed_frame.copy() if self.processed_frame is not None else None

    def _is_rapid_change(self, change_type: str) -> bool:
        """Check if this is a rapid change that should be filtered"""
        # Filter out rapid appearance/disappearance cycles
        return "gone_" in change_type or "new_" in change_type

    
    def handle_voice_command(self, frame: np.ndarray, detected_objects: list):
        """Enhanced voice command handling with detected objects"""
        try:
            # Clear visual indication
            self.audio_processor.speak_immediately("Listening... Speak your command now")
            
            # Listen for command with enhanced recognition
            command = self.audio_processor.listen_for_command(timeout=6, phrase_time_limit=6)
            
            if command:
                logger.info(f"Voice command received: '{command}'")
                
                # Process command using smart pattern matching
                action = self.audio_processor.process_voice_command(command)
                
                # Handle specific actions
                if action == "describe_scene":
                    summary = self.scene_tracker.get_scene_summary()
                    if summary and summary.strip():
                        self.audio_processor.speak_async(summary, AudioPriority.HIGH)
                        self.last_description = summary
                    else:
                        self.audio_processor.speak_immediately("I don't see anything specific right now")
                        
                        
                elif action == "emergency":
                    self.audio_processor.speak_immediately("Emergency mode activated. This feature will be available soon.")
                    
                elif action == "get_location":
                    self.audio_processor.speak_immediately("Location services will be available soon.")
                    
                elif action == "calibrate":
                    if detected_objects:  # NOW THIS WILL WORK
                        self.manual_calibration_mode(frame, detected_objects)
                    else:
                        self.audio_processor.speak_immediately("No objects visible for calibration")
                        
                elif action == "status":
                    status = self.easy_calibrator.get_calibration_status()
                    if status['is_calibrated']:
                        self.audio_processor.speak_immediately("System is running normally with calibration active")
                    else:
                        self.audio_processor.speak_immediately("System is running normally, but no calibration is active")
                        
            else:
                self.audio_processor.speak_immediately("I didn't hear any command. Please try again.")
                
        except Exception as e:
            logger.error(f"Voice command error: {e}")
            self.audio_processor.speak_immediately("Voice command failed. Please try again.")


    
    def start(self):
        """Start the NAVIS AI system"""
        if not self.start_camera():
            return False
        
        self.running = True
        self.should_exit = False
        
        # Start audio processor
        self.audio_processor.start()
        
        
        # Welcome message
        self.audio_processor.speak_immediately("Welcome to the NAVIS AI . Press S for scene description, V for voice commands and C To Calibrate Distance.")
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.run_detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        logger.info("NAVIS AI started")
        return True
    
    def stop(self):
        """Stop the NAVIS AI system"""
        self.running = False
        self.should_exit = True
        
        # Goodbye message
        self.audio_processor.speak_immediately("NAVIS AI is shutting down. Goodbye.")
        
        # Wait for thread to finish
        if self.detection_thread:
            self.detection_thread.join(timeout=5)
        
        # Stop camera
        self.stop_camera()
        
        # Stop audio
        self.audio_processor.stop()
        
        # Close windows
        cv2.destroyAllWindows()
        
        logger.info("NAVIS AI stopped")
    
    def add_known_person(self, name: str, image_path: str) -> bool:
        """Add a known person to the system"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return False
            
            return self.personalized_detector.add_known_face(name, image)
            
        except Exception as e:
            logger.error(f"Failed to add person {name}: {e}")
            return False
        
    def manual_calibration_mode(self, frame, detected_objects):
        """Enter manual calibration mode with compensation for manual scaling"""
        if not detected_objects:
            self.audio_processor.speak_immediately("No objects detected for calibration")
            return
        
        closest_obj = min(detected_objects,
                        key=lambda obj: obj.distance if obj.distance else float('inf'))
        
        self.audio_processor.speak_immediately(
            f"Ready to calibrate {closest_obj.class_name}. Press 1 for 1 meter, 2 for 2 meters, 3 for 3 meters, or 5 for 5 meters"
        )
        
        try:
            start_time = time.time()
            selected_distance = None
            
            while time.time() - start_time < 10:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('1'):
                    selected_distance = 1.0 * 2  # Multiply by 2 to compensate
                    break
                elif key == ord('2'):
                    selected_distance = 2.0 * 2  # Multiply by 2 to compensate
                    break
                elif key == ord('3'):
                    selected_distance = 3.0 * 2  # Multiply by 2 to compensate
                    break
                elif key == ord('5'):
                    selected_distance = 5.0 * 2  # Multiply by 2 to compensate
                    break
                elif key == ord('q') or key == ord('Q'):
                    self.audio_processor.speak_immediately("Calibration cancelled")
                    return
            
            if selected_distance:
                success = self.depth_estimator.add_calibration_point(
                    frame, closest_obj.bbox, selected_distance, confidence=0.9
                )
                
                actual_distance = selected_distance / 2  # Show user the actual distance
                if success:
                    self.audio_processor.speak_immediately(f"Calibration point added at {actual_distance} meters")
                    logger.info(f"Calibration successful: {closest_obj.class_name} at {actual_distance}m (compensated: {selected_distance}m)")
                else:
                    self.audio_processor.speak_immediately("Calibration failed")
            else:
                self.audio_processor.speak_immediately("Calibration timeout. No distance selected")
                
        except Exception as e:
            logger.error(f"Manual calibration failed: {e}")
            self.audio_processor.speak_immediately("Calibration error occurred")


    def show_calibration_status(self):
        """Show current calibration status"""
        try:
            status = self.easy_calibrator.get_calibration_status()
            
            if status['is_calibrated']:
                quality_text = "excellent" if status['quality_score'] > 0.8 else "good" if status['quality_score'] > 0.6 else "fair" if status['quality_score'] > 0.4 else "poor"
                message = f"Calibration is {quality_text} with {status['sample_count']} data points using {status['model_type']} model"
            else:
                message = "No calibration active. Distance estimates may be inaccurate. Press M to calibrate"
            
            self.audio_processor.speak_immediately(message)
            logger.info(f"Calibration Status: {status}")
            
            # Show recommendations
            if 'recommendations' in status and status['recommendations']:
                for rec in status['recommendations'][:2]:  # Limit to 2 recommendations
                    self.audio_processor.speak_async(rec, AudioPriority.LOW)
                    
        except Exception as e:
            logger.error(f"Failed to get calibration status: {e}")
            self.audio_processor.speak_immediately("Unable to get calibration status")

    def reset_calibration(self):
        """Reset calibration data"""
        try:
            self.depth_estimator.calibrator.reset_calibration()
            self.audio_processor.speak_immediately("Calibration data has been reset")
            logger.info("Calibration reset by user")
        except Exception as e:
            logger.error(f"Failed to reset calibration: {e}")
            self.audio_processor.speak_immediately("Failed to reset calibration")


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("Received interrupt signal")
    global vision_guide
    if vision_guide:
        vision_guide.stop()
    sys.exit(0)

def main():
    """Main function with integrated GUI"""
    app = QApplication(sys.argv)
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, lambda s, f: app.quit())
    
    # Initialize logging
    logger.add("logs/NAVIS.log", rotation="10 MB")
    
    # Create VisionGuide AI instance
    vision_guide = VisionGuideAI()
    
    try:
        # Start the vision system
        if vision_guide.start():
            logger.info("NAVIS AI started successfully")
            
            # Create and show GUI
            main_window = VisionGuideMainWindow(vision_guide)
            main_window.show()
            
            # Run the application
            sys.exit(app.exec_())
        else:
            logger.error("Failed to start NAVIS AI")
            
    except Exception as e:
        logger.error(f"Main loop error: {e}")
    finally:
        vision_guide.stop()
        logger.info("NAVIS AI shutdown complete")


if __name__ == "__main__":
    main()