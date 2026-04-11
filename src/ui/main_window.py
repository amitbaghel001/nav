import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QLabel, QFrame, QStatusBar, 
                            QGridLayout, QSizePolicy, QDialog, QTextEdit)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor
from core.audio_processor import AudioPriority

class VisionGuideMainWindow(QMainWindow):
    def __init__(self, vision_guide_ai):
        super().__init__()
        self.vision_guide_ai = vision_guide_ai
        self.init_ui()
        self.setup_timer()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("NAVIS AI - Smart Navigation Assistant")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(self.get_stylesheet())
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Camera display area
        self.setup_camera_display(main_layout)
        
        # Control panel
        self.setup_control_panel(main_layout)
        
        # Status bar
        self.setup_status_bar()
        
    def setup_camera_display(self, parent_layout):
        """Setup camera feed display area"""
        # Camera container
        camera_container = QFrame()
        camera_container.setObjectName("cameraContainer")
        camera_container.setMinimumSize(800, 600)
        
        camera_layout = QVBoxLayout(camera_container)
        
        # Title
        title_label = QLabel("Live Camera Feed")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(title_label)
        
        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setObjectName("cameraDisplay")
        self.camera_label.setMinimumSize(780, 520)
        self.camera_label.setStyleSheet("background-color: #1a1a1a; border: 2px solid #3daee9;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("Camera Starting...")
        camera_layout.addWidget(self.camera_label)
        
        # Current status
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(self.status_label)
        
        parent_layout.addWidget(camera_container, 3)
        
    def setup_control_panel(self, parent_layout):
        """Setup control panel with buttons"""
        # Control panel container
        control_panel = QFrame()
        control_panel.setObjectName("controlPanel")
        control_panel.setFixedWidth(300)
        
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(15)
        
        # App title
        app_title = QLabel("NAVIS AI")
        app_title.setObjectName("appTitle")
        app_title.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(app_title)
        
        # Subtitle
        subtitle = QLabel("Smart Navigation Assistant")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(subtitle)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setObjectName("separator")
        control_layout.addWidget(separator)
        
        # Main control buttons
        self.setup_control_buttons(control_layout)
        
        # Spacer
        control_layout.addStretch()
        
        # System info
        self.setup_system_info(control_layout)
        
        parent_layout.addWidget(control_panel, 1)
        
    def setup_control_buttons(self, layout):
        """Setup main control buttons"""
        buttons_data = [
            ("Describe Scene", "S", self.describe_scene, "#4CAF50", "🔍"),
            ("Voice Command", "V", self.voice_command, "#2196F3", "🎤"),
            ("Calibrate", "C", self.calibrate, "#FF9800", "⚙️"),
            ("Calibration Status", "N", self.calibration_status, "#9C27B0", "📊"),
            ("Reset Calibration", "X", self.reset_calibration, "#F44336", "🔄"),
            ("Quit", "Q", self.quit_app, "#607D8B", "❌")
        ]
        
        for text, shortcut, callback, color, icon in buttons_data:
            button = QPushButton(f"{icon}  {text} ({shortcut})")
            button.setObjectName("controlButton")
            button.setMinimumHeight(50)
            button.setStyleSheet(f"""
                QPushButton#controlButton {{
                    background-color: {color};
                    border: none;
                    border-radius: 8px;
                    color: white;
                    font-size: 14px;
                    font-weight: bold;
                    padding: 12px;
                }}
                QPushButton#controlButton:hover {{
                    background-color: {self.darken_color(color)};
                    transform: translateY(-2px);
                }}
                QPushButton#controlButton:pressed {{
                    background-color: {self.darken_color(color, 0.8)};
                }}
            """)
            button.clicked.connect(callback)
            layout.addWidget(button)
    
    def setup_system_info(self, layout):
        """Setup system information display"""
        info_container = QFrame()
        info_container.setObjectName("infoContainer")
        info_layout = QVBoxLayout(info_container)
        
        # System status
        self.system_status = QLabel("🟢 System Ready")
        self.system_status.setObjectName("systemStatus")
        info_layout.addWidget(self.system_status)
        
        # Calibration info
        self.calibration_info = QLabel("📏 Calibration: Not Active")
        self.calibration_info.setObjectName("calibrationInfo")
        info_layout.addWidget(self.calibration_info)
        
        # Audio status
        self.audio_status = QLabel("🔊 Audio: Ready")
        self.audio_status.setObjectName("audioStatus")
        info_layout.addWidget(self.audio_status)
        
        layout.addWidget(info_container)
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = QStatusBar()
        self.status_bar.setObjectName("statusBar")
        self.status_bar.showMessage("NAVIS AI Ready")
        self.setStatusBar(self.status_bar)
    
    def setup_timer(self):
        """Setup timer for camera updates"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(33)  # ~30 FPS
        
    def update_camera(self):
        """Update camera display with processed frame"""
        if self.vision_guide_ai.running:
            # Get processed frame from NAVIS AI
            processed_frame = self.vision_guide_ai.get_processed_frame()
            
            if processed_frame is not None:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                
                # Create QImage
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # Scale to fit label
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.camera_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                
                # Display in GUI
                self.camera_label.setPixmap(scaled_pixmap)
            else:
                # Show waiting message
                self.camera_label.setText("Processing camera feed...")

        
    def darken_color(self, color, factor=0.8):
        """Darken a hex color with better error handling"""
        try:
            # Remove # and validate
            color = color.lstrip('#')
            if len(color) != 6:
                return color  # Return original if invalid
            
            # Convert hex to RGB
            r = int(color[0:2], 16)
            g = int(color[2:4], 16)
            b = int(color[4:6], 16)
            
            # Darken each component
            r = max(0, min(255, int(r * factor)))
            g = max(0, min(255, int(g * factor)))
            b = max(0, min(255, int(b * factor)))
            
            # Convert back to hex
            return f"#{r:02x}{g:02x}{b:02x}"
            
        except ValueError:
            # Return original color if conversion fails
            return color


    
    def get_stylesheet(self):
        """Get application stylesheet"""
        return """
        QMainWindow {
            background-color: #f5f5f5;
            color: #333333;
        }
        
        #cameraContainer {
            background-color: #ffffff;
            border-radius: 12px;
            border: 1px solid #e0e0e0;
            padding: 15px;
        }
        
        #controlPanel {
            background-color: #ffffff;
            border-radius: 12px;
            border: 1px solid #e0e0e0;
            padding: 20px;
        }
        
        #titleLabel {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        #appTitle {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        
        #subtitle {
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 20px;
        }
        
        #separator {
            color: #e0e0e0;
            margin: 10px 0;
        }
        
        #statusLabel {
            font-size: 16px;
            font-weight: bold;
            color: #27ae60;
            padding: 10px;
            background-color: #ecf0f1;
            border-radius: 6px;
            margin-top: 10px;
        }
        
        #infoContainer {
            background-color: #ecf0f1;
            border-radius: 8px;
            padding: 15px;
            margin-top: 10px;
        }
        
        #systemStatus, #calibrationInfo, #audioStatus {
            font-size: 12px;
            color: #34495e;
            margin: 5px 0;
        }
        
        #statusBar {
            background-color: #34495e;
            color: #ffffff;
            padding: 5px;
        }
        """
    
    # Button callbacks
    def describe_scene(self):
        """Describe scene button callback"""
        self.status_label.setText("🔍 Describing scene...")
        self.status_bar.showMessage("Describing scene...")
        
        # Use existing NAVIS AI processing
        current_frame = self.vision_guide_ai.get_current_frame()
        if current_frame is not None:
            audio_description = self.vision_guide_ai.process_frame(current_frame)
            if audio_description and audio_description.strip() != ".":
                self.vision_guide_ai.audio_processor.speak_async(audio_description, AudioPriority.HIGH)
                self.status_label.setText("✅ Scene described")
            else:
                self.status_label.setText("❌ No scene to describe")



    def voice_command(self):
        """Voice command button callback"""
        self.status_label.setText("🎤 Listening for command...")
        self.status_bar.showMessage("Voice command mode active")
        
        # Use existing voice command handler
        current_frame = self.vision_guide_ai.get_current_frame()
        detected_objects = []  # You can get this from your vision system if needed
        
        if current_frame is not None:
            self.vision_guide_ai.handle_voice_command(current_frame, detected_objects)
        
        self.status_label.setText("✅ Voice command processed")

    # Remove keyboard event handling since we don't need OpenCV window keys
    # def keyPressEvent(self, event):
    #     # Remove this method entirely

        
        
    def calibrate(self):
        """Calibrate button callback"""
        self.status_label.setText("⚙️ Starting calibration...")
        self.status_bar.showMessage("Calibration mode")
        # Call your existing calibration logic here
        
    def calibration_status(self):
        """Show calibration status"""
        dialog = CalibrationStatusDialog(self)
        dialog.exec_()
        
    def reset_calibration(self):
        """Reset calibration"""
        self.status_label.setText("🔄 Resetting calibration...")
        self.status_bar.showMessage("Calibration reset")
        # Call your existing reset logic here
        
    def quit_app(self):
        """Quit application"""
        self.vision_guide_ai.stop()
        QApplication.quit()
        
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        key_map = {
            Qt.Key_S: self.describe_scene,
            Qt.Key_V: self.voice_command,
            Qt.Key_C: self.calibrate,
            Qt.Key_N: self.calibration_status,
            Qt.Key_X: self.reset_calibration,
            Qt.Key_Q: self.quit_app
        }
        
        if event.key() in key_map:
            key_map[event.key()]()
        else:
            super().keyPressEvent(event)

class CalibrationStatusDialog(QDialog):
    """Dialog for showing calibration status"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize dialog UI"""
        self.setWindowTitle("Calibration Status")
        self.setFixedSize(400, 300)
        self.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
                border-radius: 8px;
            }
            QLabel {
                color: #2c3e50;
                font-size: 14px;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("📊 Calibration Status")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 15px;")
        layout.addWidget(title)
        
        # Status info
        status_info = QTextEdit()
        status_info.setReadOnly(True)
        status_info.setPlainText("""
Current Status: Not Calibrated
Data Points: 0
Model Type: None
Quality Score: N/A

Recommendations:
• Add calibration points for better accuracy
• Calibrate with objects at different distances
        """)
        layout.addWidget(status_info)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
