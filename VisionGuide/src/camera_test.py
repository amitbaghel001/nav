import cv2
import time
import psutil
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from app import VisionGuideAI
from utils.config import config

def test_performance():
    """Test system performance"""
    print("Starting VisionGuide AI Performance Test...")
    
    # Initialize system
    vision_guide = VisionGuideAI()
    
    if not vision_guide.start_camera():
        print("❌ Failed to start camera")
        return False
    
    start_time = time.time()
    frame_count = 0
    test_frames = 100  # Test 100 frames
    
    print(f"Testing {test_frames} frames...")
    
    try:
        while frame_count < test_frames:
            ret, frame = vision_guide.camera.read()
            if ret:
                # Simulate processing
                detected_objects = vision_guide.object_detector.detect_objects(
                    frame, 
                    config.model.confidence_threshold,
                    config.model.iou_threshold
                )
                frame_count += 1
                
                # Show progress
                if frame_count % 20 == 0:
                    print(f"Processed {frame_count}/{test_frames} frames...")
            else:
                print("❌ Failed to read frame")
                break
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_fps = frame_count / total_time
        
        # Get system metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        
        # Results
        print("\n" + "="*50)
        print("PERFORMANCE TEST RESULTS")
        print("="*50)
        print(f"✅ Frames Processed: {frame_count}")
        print(f"✅ Total Time: {total_time:.2f} seconds")
        print(f"✅ Average FPS: {avg_fps:.2f}")
        print(f"✅ CPU Usage: {cpu_usage:.1f}%")
        print(f"✅ Memory Usage: {memory_usage:.1f}%")
        print(f"✅ Available Memory: {memory_info.available / (1024**3):.1f} GB")
        
        # Performance evaluation
        if avg_fps >= 15:
            print("🎉 EXCELLENT: FPS is optimal!")
        elif avg_fps >= 10:
            print("✅ GOOD: FPS is acceptable")
        elif avg_fps >= 5:
            print("⚠️  MODERATE: FPS needs improvement")
        else:
            print("❌ POOR: FPS is too low")
        
        return avg_fps >= 10
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        vision_guide.stop_camera()
        print("\nTest completed!")

def test_camera_only():
    """Test camera performance without AI processing"""
    print("\nTesting camera-only performance...")
    
    camera = cv2.VideoCapture(config.camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
    
    start_time = time.time()
    frame_count = 0
    test_frames = 100
    
    while frame_count < test_frames:
        ret, frame = camera.read()
        if ret:
            frame_count += 1
        else:
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    camera_fps = frame_count / total_time
    
    camera.release()
    
    print(f"Camera-only FPS: {camera_fps:.2f}")
    return camera_fps

if __name__ == "__main__":
    print("VisionGuide AI Performance Testing Tool")
    print("="*40)
    
    # Test camera first
    camera_fps = test_camera_only()
    
    # Test full system
    system_performance = test_performance()
    
    print(f"\nFINAL SUMMARY:")
    print(f"Camera FPS: {camera_fps:.2f}")
    print(f"System Performance: {'PASS' if system_performance else 'NEEDS OPTIMIZATION'}")
