import pyttsx3
import speech_recognition as sr
import threading
import queue
import time
import subprocess
import re
from typing import Optional, Callable, Dict, List
from loguru import logger
from dataclasses import dataclass
from enum import Enum

class AudioPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    EMERGENCY = 4

@dataclass
class SpeechRequest:
    text: str
    priority: AudioPriority
    timestamp: float
    immediate: bool = False
    callback: Optional[Callable] = None

class SmartVoiceCommandProcessor:
    """Voice command processor with pattern matching"""
    
    def __init__(self):
        self.command_patterns = {
            'describe_scene': [
        r'what do you see',
        r'describe scene',
        r'tell me what\'s here',
        r'what\'s in front of me',
        r'what\'s around me',
        r'describe surroundings',
        r'what\'s there',
        r'what can you see',
        r'describe what\'s here',
        r'tell me about the scene',
        r'what objects are here',
        r'what\'s nearby',
        r'scan the area',
        r'look around',
        r'what\'s present',
        r'identify objects',
        r'what am i looking at',
        r'analyze scene',
        r'what\'s visible',
        r'give me details'
    ],
    
    # Navigation commands
    'repeat_last': [
        r'repeat',
        r'say again',
        r'say that again',
        r'repeat last',
        r'what did you say',
        r'could you repeat',
        r'pardon',
        r'excuse me',
        r'i didn\'t catch that',
        r'come again',
        r'repeat that',
        r'say it again',
        r'one more time',
        r'didn\'t hear you',
        r'missed that'
    ],
    
    # Control commands
    'stop_talking': [
        r'stop talking',
        r'be quiet',
        r'silence',
        r'shut up',
        r'stop speaking',
        r'mute',
        r'quiet',
        r'hush',
        r'stop',
        r'enough',
        r'pause',
        r'stop now',
        r'be silent',
        r'no more',
        r'that\'s enough'
    ],
    
    # Emergency commands
    'emergency': [
        r'help me',
        r'emergency',
        r'call for help',
        r'get help',
        r'i need help',
        r'assistance',
        r'urgent',
        r'crisis',
        r'emergency help',
        r'call emergency',
        r'need assistance',
        r'help please',
        r'emergency situation',
        r'urgent help',
        r'immediate help',
        r'panic',
        r'danger',
        r'trouble',
        r'call someone',
        r'get someone'
    ],
    
    # Location commands
    'get_location': [
        r'where am i',
        r'what\'s my location',
        r'where is this',
        r'current location',
        r'my position',
        r'where are we',
        r'what place is this',
        r'location please',
        r'tell me location',
        r'find my location',
        r'gps location',
        r'coordinates',
        r'address',
        r'what\'s the address',
        r'where exactly am i'
    ],
    
    # Audio control commands
    'volume_up': [
        r'volume up',
        r'louder',
        r'increase volume',
        r'turn up',
        r'speak louder',
        r'higher volume',
        r'boost volume',
        r'make it louder',
        r'turn volume up',
        r'increase sound',
        r'more volume',
        r'amplify'
    ],
    
    'volume_down': [
        r'volume down',
        r'quieter',
        r'decrease volume',
        r'turn down',
        r'speak quieter',
        r'lower volume',
        r'reduce volume',
        r'make it quieter',
        r'turn volume down',
        r'decrease sound',
        r'less volume',
        r'softer'
    ],
    
    'speed_up': [
        r'speak faster',
        r'speed up',
        r'talk faster',
        r'faster speech',
        r'increase speed',
        r'talk quickly',
        r'speak quickly',
        r'more speed',
        r'accelerate speech',
        r'quick speech'
    ],
    
    'slow_down': [
        r'speak slower',
        r'slow down',
        r'talk slower',
        r'slower speech',
        r'decrease speed',
        r'talk slowly',
        r'speak slowly',
        r'less speed',
        r'decelerate speech',
        r'slow speech'
    ],
    
    # Calibration commands
    'calibrate': [
        r'calibrate',
        r'calibration',
        r'adjust distance',
        r'fix distance',
        r'calibrate distance',
        r'distance calibration',
        r'adjust measurements',
        r'fix measurements',
        r'tune distance',
        r'correct distance',
        r'set distance',
        r'configure distance',
        r'distance setup',
        r'measurement setup'
    ],
    
    # System commands
    'status': [
        r'system status',
        r'how are you',
        r'status report',
        r'check status',
        r'system check',
        r'health check',
        r'are you working',
        r'system info',
        r'diagnostic',
        r'report status',
        r'how\'s the system',
        r'everything ok',
        r'system report',
        r'check system'
    ]
        }
        
        self.compiled_patterns = {}
        for command, patterns in self.command_patterns.items():
            self.compiled_patterns[command] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def process_command(self, text: str) -> str:
        if not text:
            return "unknown"
        
        text = text.strip().lower()
        for command, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    logger.info(f"Matched command '{command}' from text: '{text}'")
                    return command
        return "unknown"

class UnifiedSpeechManager:
    """Unified speech manager - ONLY ONE SPEECH AT A TIME"""
    
    def __init__(self):
        # Single speech queue for ALL speech
        self.speech_queue = queue.Queue()
        
        # Global speech state
        self.is_speaking = False
        self.current_process = None
        self.speech_thread = None
        self.running = False
        
        # Master speech lock
        self._speech_lock = threading.RLock()
        
        logger.info("Unified speech manager initialized")
    
    def add_speech(self, text: str, priority: AudioPriority = AudioPriority.NORMAL, 
                   immediate: bool = False, callback: Optional[Callable] = None):
        """Add speech to unified queue"""
        if not text or text.strip() == "." or text.strip() == "":
            return
        
        request = SpeechRequest(
            text=text,
            priority=priority,
            timestamp=time.time(),
            immediate=immediate,
            callback=callback
        )
        
        with self._speech_lock:
            if immediate:
                # Clear queue and add immediate request at front
                self._clear_non_emergency_queue()
                # Create new queue with immediate request first
                temp_queue = queue.Queue()
                temp_queue.put(request)
                
                # Add back remaining items
                while not self.speech_queue.empty():
                    try:
                        temp_queue.put(self.speech_queue.get_nowait())
                    except queue.Empty:
                        break
                
                self.speech_queue = temp_queue
                logger.info(f"Immediate speech queued: {text[:50]}...")
            else:
                self.speech_queue.put(request)
                logger.debug(f"Normal speech queued: {text[:50]}...")
    
    def stop_all_speech(self):
        """Stop all speech immediately"""
        with self._speech_lock:
            logger.info("STOPPING ALL SPEECH")
            
            # Terminate current process if running
            if self.current_process:
                try:
                    self.current_process.terminate()
                    self.current_process.wait(timeout=2)
                except:
                    pass
                self.current_process = None
            
            # Clear entire queue
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.is_speaking = False
            logger.info("All speech stopped and queue cleared")
    
    def _clear_non_emergency_queue(self):
        """Clear non-emergency items from queue"""
        temp_queue = queue.Queue()
        
        while not self.speech_queue.empty():
            try:
                request = self.speech_queue.get_nowait()
                if request.priority == AudioPriority.EMERGENCY:
                    temp_queue.put(request)
            except queue.Empty:
                break
        
        self.speech_queue = temp_queue
    
    def _speak_with_sapi(self, text: str) -> bool:
        """Speak using Windows SAPI - blocking"""
        try:
            escaped_text = text.replace('"', '""').replace("'", "''")
            
            ps_command = f'''
            Add-Type -AssemblyName System.Speech
            $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
            $synth.Rate = 0
            $synth.Volume = 100
            $synth.Speak("{escaped_text}")
            '''
            
            self.current_process = subprocess.Popen([
                'powershell', '-Command', ps_command
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            stdout, stderr = self.current_process.communicate(timeout=30)
            success = self.current_process.returncode == 0
            self.current_process = None
            
            return success
            
        except Exception as e:
            logger.error(f"SAPI error: {e}")
            if self.current_process:
                try:
                    self.current_process.terminate()
                except:
                    pass
                self.current_process = None
            return False
    
    def _speech_worker(self):
        """Single speech worker - processes ALL speech sequentially"""
        logger.info("Unified speech worker started")
        
        while self.running:
            try:
                # Get next speech request
                try:
                    request = self.speech_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Skip old messages (except emergency)
                if (request.priority != AudioPriority.EMERGENCY and 
                    time.time() - request.timestamp > 20):
                    continue
                
                # Process speech request
                with self._speech_lock:
                    self.is_speaking = True
                    logger.info(f"SPEAKING: {request.text}")
                    
                    success = self._speak_with_sapi(request.text)
                    
                    if success:
                        logger.info("Speech completed successfully")
                    else:
                        logger.warning("Speech failed")
                    
                    self.is_speaking = False
                
                # Execute callback
                if request.callback:
                    try:
                        request.callback()
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                # Small pause between speeches
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Speech worker error: {e}")
                self.is_speaking = False
                time.sleep(0.5)
        
        logger.info("Unified speech worker stopped")
    
    def start(self):
        """Start speech manager"""
        if self.running:
            return
        
        self.running = True
        self.speech_thread = threading.Thread(target=self._speech_worker)
        self.speech_thread.daemon = True
        self.speech_thread.start()
        
        logger.info("Unified speech manager started")
    
    def stop(self):
        """Stop speech manager"""
        self.running = False
        self.stop_all_speech()
        
        if self.speech_thread:
            self.speech_thread.join(timeout=5)
        
        logger.info("Unified speech manager stopped")

class AudioProcessor:
    """Simplified audio processor using unified speech manager"""
    
    def __init__(self):
        # Speech management
        self.speech_manager = UnifiedSpeechManager()
        
        # Voice recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.command_processor = SmartVoiceCommandProcessor()
        
        # Voice recognition lock
        self._listening_lock = threading.Lock()
        self._is_listening = False
        
        # Configure recognizer
        self._configure_recognizer()
        self._initialize_speech_recognition()
        
        logger.info("Simplified audio processor initialized")
    
    def _configure_recognizer(self):
        """Configure speech recognizer"""
        self.recognizer.energy_threshold = 350
        self.recognizer.pause_threshold = 0.8
        self.recognizer.dynamic_energy_adjustment = True
    
    def _initialize_speech_recognition(self):
        """Initialize speech recognition"""
        try:
            with self.microphone as source:
                logger.info("Calibrating microphone...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            logger.info("Speech recognition ready")
        except Exception as e:
            logger.error(f"Speech recognition init failed: {e}")
    
    def speak_async(self, text: str, priority: AudioPriority = AudioPriority.NORMAL):
        """Add speech to queue"""
        self.speech_manager.add_speech(text, priority, immediate=False)
    
    def speak_immediately(self, text: str):
        """Speak immediately (interrupts queue)"""
        self.speech_manager.add_speech(text, AudioPriority.HIGH, immediate=True)
    
    def stop_speaking(self):
        """Stop all speech"""
        self.speech_manager.stop_all_speech()
    
    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        return self.speech_manager.is_speaking
    
    def listen_for_command(self, timeout: int = 6, phrase_time_limit: int = 6) -> str:
        """Listen for voice command"""
        if not self._listening_lock.acquire(blocking=False):
            logger.warning("Already listening")
            return ""
        
        try:
            self._is_listening = True
            logger.info("Starting voice recognition...")
            
            for attempt in range(2):
                try:
                    with self.microphone as source:
                        logger.info(f"Attempt {attempt + 1}: Listening...")
                        self.recognizer.adjust_for_ambient_noise(source, duration=1)
                        audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)  # CORRECT PARAMETER
                    
                    logger.info("Processing speech...")
                    command = self.recognizer.recognize_google(audio, language='en-US')
                    
                    if command and len(command.strip()) > 1:
                        logger.info(f"Recognized: '{command}'")
                        return command.lower().strip()
                
                except (sr.UnknownValueError, sr.WaitTimeoutError):
                    logger.info(f"Attempt {attempt + 1}: No speech detected")
                    if attempt == 0:
                        continue
                except sr.RequestError as e:
                    logger.error(f"Recognition error: {e}")
                    break
            
            return ""
            
        finally:
            self._is_listening = False
            self._listening_lock.release()

    
    def process_voice_command(self, command: str) -> str:
        """Process voice command"""
        if not command:
            return "unknown"
        
        action = self.command_processor.process_command(command)
        
        # Process commands with immediate responses
        if action == "describe_scene":
            self.speak_immediately("Getting scene description")
            
        elif action == "stop_talking":
            self.stop_speaking()
            self.speak_immediately("Speech stopped")
            
        elif action == "repeat_last":
            self.speak_immediately("Repeating last message")
            
        elif action == "emergency":
            self.speak_immediately("Emergency mode activated")
            
        elif action == "get_location":
            self.speak_immediately("Getting location")
            
        elif action == "volume_up":
            self.speak_immediately("Volume increased")
            
        elif action == "volume_down":
            self.speak_immediately("Volume decreased")
            
        elif action == "status":
            self.speak_immediately("System running normally")
            
        elif action == "unknown":
            self.speak_immediately("Command not recognized")
        
        return action
    
    def start(self):
        """Start audio processor"""
        self.speech_manager.start()
        logger.info("Audio processor started")
    
    def stop(self):
        """Stop audio processor"""
        self.speech_manager.stop()
        logger.info("Audio processor stopped")
    
    def test_audio(self):
        """Test audio"""
        self.speak_immediately("VisionGuide AI audio test. Sequential speech working correctly.")