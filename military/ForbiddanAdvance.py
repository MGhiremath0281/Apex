import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import threading
import pygame
from datetime import datetime
import logging
from typing import List, Tuple, Dict, Optional
import math
import json
import os
from collections import deque
import sqlite3
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns

@dataclass
class Detection:
    """Data class for detection information"""
    timestamp: float
    object_class: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[int, int]
    zone_id: Optional[int] = None
    is_intrusion: bool = False

class AdvancedIntrusionDetectionSystem:
    def __init__(self, model_path: str = 'yolov8x.pt', source: int = 0):
        """
        Initialize the Advanced Intrusion Detection System with enhanced features
        """
        # Initialize YOLO model
        self.model = YOLO(model_path)
        
        # Video capture with enhanced settings
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get frame dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize multiple forbidden zones with different shapes
        self.zones = {
            1: self._create_pentagon_zone(0.15, 0.15, 0.35, 0.35),
            2: self._create_hexagon_zone(0.65, 0.65, 0.85, 0.85),
            3: self._create_octagon_zone(0.4, 0.1, 0.6, 0.3),
            4: self._create_triangle_zone(0.1, 0.6, 0.3, 0.9)
        }
        
        # Zone properties
        self.zone_colors = {
            1: (0, 0, 255),      # Red
            2: (255, 0, 0),      # Blue
            3: (0, 255, 0),      # Green
            4: (255, 0, 255)     # Magenta
        }
        
        self.zone_names = {
            1: "CRITICAL ZONE",
            2: "HIGH SECURITY",
            3: "RESTRICTED AREA",
            4: "PERIMETER"
        }
        
        self.zone_enabled = {1: True, 2: True, 3: True, 4: True}
        self.zone_sensitivity = {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.6}
        
        # Enhanced alert system
        self.alert_active = False
        self.alert_start_time = 0
        self.alert_duration = 5.0
        self.alert_zones = set()
        self.alert_history = deque(maxlen=100)
        
        # Database for logging
        self.init_database()
        
        # Audio system with multiple sounds
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        self.alert_sounds = {
            'critical': self._create_alert_sound(1000, 0.8),
            'warning': self._create_alert_sound(800, 0.6),
            'info': self._create_alert_sound(600, 0.4)
        }
        
        # Enhanced object tracking
        self.target_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
            5: 'bus', 7: 'truck', 14: 'bird', 15: 'cat', 16: 'dog',
            17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
            21: 'bear', 22: 'zebra', 23: 'giraffe'
        }
        
        # Object tracking and trail system
        self.object_trails = {}
        self.trail_length = 30
        self.object_id_counter = 0
        
        # Zone adjustment parameters
        self.zone_resize_factor = 0.05
        self.selected_zone = 1
        self.zone_rotation_angle = 0
        
        # Performance and analytics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.detection_count = 0
        self.intrusion_count = 0
        self.session_start = time.time()
        
        # Advanced visualization
        self.show_heatmap = False
        self.show_analytics = False
        self.show_trails = True
        self.show_confidence = True
        self.night_mode = False
        
        # Detection history for heatmap
        self.detection_heatmap = np.zeros((self.frame_height // 10, self.frame_width // 10), dtype=np.float32)
        
        # Statistics tracking
        self.stats = {
            'total_detections': 0,
            'total_intrusions': 0,
            'zone_intrusions': {1: 0, 2: 0, 3: 0, 4: 0},
            'class_detections': {cls: 0 for cls in self.target_classes.values()}
        }
        
        # Recording system
        self.recording = False
        self.video_writer = None
        
        print("🚀 Advanced Intrusion Detection System Initialized")
        print("=" * 60)
        self._print_controls()
    
    def _print_controls(self):
        """Print enhanced control instructions"""
        controls = [
            "ZONE CONTROLS:",
            "  1/2/3/4 - Select Zone",
            "  + - Increase zone size",
            "  - - Decrease zone size", 
            "  r - Rotate selected zone",
            "  t - Toggle selected zone",
            "  SHIFT+R - Reset all zones",
            "",
            "VISUALIZATION:",
            "  h - Toggle heatmap",
            "  a - Toggle analytics panel",
            "  v - Toggle object trails",
            "  c - Toggle confidence display",
            "  n - Toggle night mode",
            "",
            "SYSTEM:",
            "  s - Save configuration",
            "  l - Load configuration",
            "  SPACE - Start/Stop recording",
            "  ESC - Emergency stop all alerts",
            "  q - Quit system"
        ]
        
        for control in controls:
            print(control)
        print("=" * 60)
    
    def init_database(self):
        """Initialize SQLite database for logging"""
        self.db_path = 'intrusion_detection.db'
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                object_class TEXT,
                confidence REAL,
                zone_id INTEGER,
                is_intrusion BOOLEAN,
                centroid_x INTEGER,
                centroid_y INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                zone_id INTEGER,
                object_class TEXT,
                severity TEXT,
                resolved_timestamp REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _create_pentagon_zone(self, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        """Create a pentagon-shaped zone"""
        center_x = (x1 + x2) / 2 * self.frame_width
        center_y = (y1 + y2) / 2 * self.frame_height
        radius = min((x2 - x1) * self.frame_width, (y2 - y1) * self.frame_height) / 2
        
        points = []
        for i in range(5):
            angle = 2 * math.pi * i / 5 - math.pi / 2
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append([int(x), int(y)])
        
        return np.array(points, dtype=np.int32)
    
    def _create_hexagon_zone(self, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        """Create a hexagon-shaped zone"""
        center_x = (x1 + x2) / 2 * self.frame_width
        center_y = (y1 + y2) / 2 * self.frame_height
        radius = min((x2 - x1) * self.frame_width, (y2 - y1) * self.frame_height) / 2
        
        points = []
        for i in range(6):
            angle = 2 * math.pi * i / 6
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append([int(x), int(y)])
        
        return np.array(points, dtype=np.int32)
    
    def _create_octagon_zone(self, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        """Create an octagon-shaped zone"""
        center_x = (x1 + x2) / 2 * self.frame_width
        center_y = (y1 + y2) / 2 * self.frame_height
        radius = min((x2 - x1) * self.frame_width, (y2 - y1) * self.frame_height) / 2
        
        points = []
        for i in range(8):
            angle = 2 * math.pi * i / 8
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append([int(x), int(y)])
        
        return np.array(points, dtype=np.int32)
    
    def _create_triangle_zone(self, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        """Create a triangle-shaped zone"""
        center_x = (x1 + x2) / 2 * self.frame_width
        center_y = (y1 + y2) / 2 * self.frame_height
        radius = min((x2 - x1) * self.frame_width, (y2 - y1) * self.frame_height) / 2
        
        points = []
        for i in range(3):
            angle = 2 * math.pi * i / 3 - math.pi / 2
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append([int(x), int(y)])
        
        return np.array(points, dtype=np.int32)
    
    def _create_alert_sound(self, frequency: int, duration: float) -> pygame.mixer.Sound:
        """Create enhanced alert sounds"""
        sample_rate = 22050
        frames = int(duration * sample_rate)
        
        # Create a more complex waveform
        t = np.linspace(0, duration, frames)
        wave = np.sin(2 * np.pi * frequency * t) * np.exp(-t * 2)  # Exponential decay
        wave += 0.3 * np.sin(2 * np.pi * frequency * 2 * t) * np.exp(-t * 3)  # Harmonic
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.05, frames)
        wave += noise
        
        # Normalize and convert to 16-bit
        wave = np.clip(wave, -1, 1)
        wave = (wave * 32767).astype(np.int16)
        
        # Create stereo
        stereo = np.zeros((frames, 2), dtype=np.int16)
        stereo[:, 0] = wave
        stereo[:, 1] = wave
        
        return pygame.sndarray.make_sound(stereo)
    
    def point_in_polygon(self, point: Tuple[int, int], polygon: np.ndarray) -> bool:
        """Enhanced point-in-polygon with edge case handling"""
        if polygon is None or len(polygon) < 3:
            return False
            
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def rotate_zone(self, zone_id: int, angle: float):
        """Rotate a zone around its center"""
        if zone_id not in self.zones:
            return
            
        zone = self.zones[zone_id]
        center_x = np.mean(zone[:, 0])
        center_y = np.mean(zone[:, 1])
        
        # Rotation matrix
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        
        rotated_zone = zone.copy()
        for i in range(len(rotated_zone)):
            # Translate to origin
            x = rotated_zone[i, 0] - center_x
            y = rotated_zone[i, 1] - center_y
            
            # Rotate
            new_x = x * cos_angle - y * sin_angle
            new_y = x * sin_angle + y * cos_angle
            
            # Translate back
            rotated_zone[i, 0] = int(new_x + center_x)
            rotated_zone[i, 1] = int(new_y + center_y)
        
        self.zones[zone_id] = rotated_zone
        print(f"Zone {zone_id} rotated by {math.degrees(angle):.1f}°")
    
    def resize_zone(self, zone_id: int, increase: bool):
        """Enhanced zone resizing with boundary checking"""
        if zone_id not in self.zones:
            return
            
        zone = self.zones[zone_id]
        center_x = np.mean(zone[:, 0])
        center_y = np.mean(zone[:, 1])
        
        # Dynamic scale factor based on zone size
        current_size = np.mean(np.sqrt(np.sum((zone - [center_x, center_y])**2, axis=1)))
        adaptive_factor = max(0.02, min(0.1, self.zone_resize_factor * (100 / current_size)))
        
        scale = 1 + adaptive_factor if increase else 1 - adaptive_factor
        
        # Prevent zones from becoming too small or too large
        if not increase and current_size < 30:
            print(f"Zone {zone_id} cannot be made smaller")
            return
        if increase and current_size > 300:
            print(f"Zone {zone_id} cannot be made larger")
            return
        
        scaled_zone = zone.copy()
        for i in range(len(scaled_zone)):
            dx = scaled_zone[i, 0] - center_x
            dy = scaled_zone[i, 1] - center_y
            scaled_zone[i, 0] = int(center_x + dx * scale)
            scaled_zone[i, 1] = int(center_y + dy * scale)
        
        # Boundary checking
        scaled_zone[:, 0] = np.clip(scaled_zone[:, 0], 0, self.frame_width)
        scaled_zone[:, 1] = np.clip(scaled_zone[:, 1], 0, self.frame_height)
        
        self.zones[zone_id] = scaled_zone
        print(f"Zone {zone_id} {'increased' if increase else 'decreased'} (size: {current_size:.1f})")
    
    def get_zone_priority(self, zone_id: int) -> str:
        """Get zone priority level"""
        priorities = {1: 'CRITICAL', 2: 'HIGH', 3: 'MEDIUM', 4: 'LOW'}
        return priorities.get(zone_id, 'UNKNOWN')
    
    def trigger_alert(self, zone_id: int, object_class: str, centroid: Tuple[int, int], confidence: float):
        """Enhanced alert system with priority levels"""
        if zone_id not in self.zones or not self.zone_enabled[zone_id]:
            return
            
        priority = self.get_zone_priority(zone_id)
        
        # Check if confidence meets zone sensitivity
        if confidence < self.zone_sensitivity[zone_id]:
            return
        
        self.alert_active = True
        self.alert_start_time = time.time()
        self.alert_zones.add(zone_id)
        
        # Create alert record
        alert_record = {
            'timestamp': time.time(),
            'zone_id': zone_id,
            'object_class': object_class,
            'priority': priority,
            'confidence': confidence,
            'centroid': centroid
        }
        
        self.alert_history.append(alert_record)
        
        # Log to database
        self._log_to_database(alert_record)
        
        # Play appropriate sound
        if priority == 'CRITICAL':
            self.alert_sounds['critical'].play()
        elif priority == 'HIGH':
            self.alert_sounds['warning'].play()
        else:
            self.alert_sounds['info'].play()
        
        # Update statistics
        self.stats['total_intrusions'] += 1
        self.stats['zone_intrusions'][zone_id] += 1
        
        log_message = f"🚨 {priority} ALERT - Zone {zone_id} ({self.zone_names[zone_id]}), Object: {object_class}, Confidence: {confidence:.2f}, Position: {centroid}"
        logging.warning(log_message)
        print(log_message)
    
    def _log_to_database(self, alert_record: Dict):
        """Log alert to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts (timestamp, zone_id, object_class, severity)
                VALUES (?, ?, ?, ?)
            ''', (alert_record['timestamp'], alert_record['zone_id'], 
                  alert_record['object_class'], alert_record['priority']))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database error: {e}")
    
    def update_heatmap(self, centroid: Tuple[int, int]):
        """Update detection heatmap"""
        x, y = centroid
        heat_x = min(x // 10, self.detection_heatmap.shape[1] - 1)
        heat_y = min(y // 10, self.detection_heatmap.shape[0] - 1)
        
        # Add gaussian blur effect
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                hx = max(0, min(heat_x + dx, self.detection_heatmap.shape[1] - 1))
                hy = max(0, min(heat_y + dy, self.detection_heatmap.shape[0] - 1))
                distance = math.sqrt(dx*dx + dy*dy)
                intensity = math.exp(-distance * 0.5)
                self.detection_heatmap[hy, hx] += intensity
        
        # Decay over time
        self.detection_heatmap *= 0.995
    
    def draw_enhanced_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw zones with enhanced visualization"""
        overlay = frame.copy()
        
        for zone_id, zone in self.zones.items():
            if not self.zone_enabled[zone_id]:
                continue
                
            color = self.zone_colors[zone_id]
            
            # Animated border for selected zone
            if zone_id == self.selected_zone:
                border_width = 5 + int(3 * math.sin(time.time() * 10))
                border_color = tuple(int(c * 0.8) for c in color)
            else:
                border_width = 2
                border_color = color
            
            # Fill zone with transparency
            cv2.fillPoly(overlay, [zone], color)
            
            # Draw border
            cv2.polylines(frame, [zone], True, border_color, border_width)
            
            # Add zone info
            center_x = int(np.mean(zone[:, 0]))
            center_y = int(np.mean(zone[:, 1]))
            
            zone_info = [
                f"ZONE {zone_id}",
                self.zone_names[zone_id],
                f"Priority: {self.get_zone_priority(zone_id)}",
                f"Sensitivity: {self.zone_sensitivity[zone_id]:.1f}"
            ]
            
            if zone_id == self.selected_zone:
                zone_info.append("(SELECTED)")
            
            for i, info in enumerate(zone_info):
                text_y = center_y - 30 + i * 15
                cv2.putText(frame, info, (center_x - 60, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Blend with transparency
        alpha = 0.3 if not self.night_mode else 0.5
        frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        return frame
    
    def draw_heatmap(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection heatmap overlay"""
        if not self.show_heatmap:
            return frame
            
        # Resize heatmap to frame size
        heatmap_resized = cv2.resize(self.detection_heatmap, (self.frame_width, self.frame_height))
        
        # Normalize and apply colormap
        heatmap_norm = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with frame
        frame = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)
        
        return frame
    
    def draw_analytics_panel(self, frame: np.ndarray) -> np.ndarray:
        """Draw advanced analytics panel"""
        if not self.show_analytics:
            return frame
            
        panel_width = 300
        panel_height = 400
        panel_x = frame.shape[1] - panel_width - 10
        panel_y = 10
        
        # Create semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Panel content
        session_time = time.time() - self.session_start
        
        analytics = [
            " SYSTEM ANALYTICS",
            f"Session: {int(session_time//3600):02d}:{int((session_time%3600)//60):02d}:{int(session_time%60):02d}",
            f"FPS: {self.current_fps}",
            f"Total Detections: {self.stats['total_detections']}",
            f"Total Intrusions: {self.stats['total_intrusions']}",
            "",
            " ZONE STATISTICS:",
            f"Zone 1: {self.stats['zone_intrusions'][1]} intrusions",
            f"Zone 2: {self.stats['zone_intrusions'][2]} intrusions", 
            f"Zone 3: {self.stats['zone_intrusions'][3]} intrusions",
            f"Zone 4: {self.stats['zone_intrusions'][4]} intrusions",
            "",
            " TOP DETECTIONS:",
        ]
        
        # Add top detected classes
        sorted_classes = sorted(self.stats['class_detections'].items(), 
                              key=lambda x: x[1], reverse=True)[:5]
        for class_name, count in sorted_classes:
            if count > 0:
                analytics.append(f"{class_name}: {count}")
        
        # Recent alerts
        if self.alert_history:
            analytics.extend(["", "🚨 RECENT ALERTS:"])
            for alert in list(self.alert_history)[-3:]:
                time_str = datetime.fromtimestamp(alert['timestamp']).strftime("%H:%M:%S")
                analytics.append(f"{time_str} - Zone {alert['zone_id']}: {alert['object_class']}")
        
        # Draw analytics text
        for i, text in enumerate(analytics):
            text_y = panel_y + 20 + i * 18
            if text_y > panel_y + panel_height - 20:
                break
                
            color = (0, 255, 255) if text.startswith(('🔍', '📊', '🎯', '🚨')) else (255, 255, 255)
            cv2.putText(frame, text, (panel_x + 10, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def draw_object_trails(self, frame: np.ndarray) -> np.ndarray:
        """Draw object movement trails"""
        if not self.show_trails:
            return frame
            
        for obj_id, trail in self.object_trails.items():
            if len(trail) > 1:
                # Draw trail with fading effect
                for i in range(1, len(trail)):
                    alpha = i / len(trail)
                    thickness = max(1, int(3 * alpha))
                    color = (int(255 * alpha), int(255 * alpha), 0)
                    
                    cv2.line(frame, trail[i-1], trail[i], color, thickness)
        
        return frame
    
    def draw_enhanced_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw enhanced UI elements"""
        # Apply night mode filter
        if self.night_mode:
            frame = cv2.convertScaleAbs(frame, alpha=0.7, beta=30)
        
        # Status bar
        status_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], status_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Main status information
        session_time = time.time() - self.session_start
        status_text = [
            f" ADVANCED INTRUSION DETECTION | FPS: {self.current_fps} | "
            f"Session: {int(session_time//3600):02d}:{int((session_time%3600)//60):02d}:{int(session_time%60):02d}",
            f"Selected Zone: {self.selected_zone} ({self.zone_names[self.selected_zone]}) | "
            f"Active Zones: {sum(self.zone_enabled.values())} | "
            f"Recording: {'ON' if self.recording else 'OFF'}"
        ]
        
        for i, text in enumerate(status_text):
            cv2.putText(frame, text, (10, 25 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Quick controls reminder
        controls_text = "Controls: +/- (resize) | r (rotate) | h (heatmap) | a (analytics) | SPACE (record)"
        cv2.putText(frame, controls_text, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_enhanced_alert_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw enhanced alert overlay with multiple effects"""
        if not self.alert_active:
            return frame
            
        elapsed = time.time() - self.alert_start_time
        
        if elapsed < self.alert_duration:
            # Pulsing effect
            pulse_intensity = 0.3 + 0.2 * math.sin(elapsed * 15)
            
            # Multi-color alert based on zones
            alert_color = (0, 0, 255)  # Default red
            if 1 in self.alert_zones:  # Critical zone
                alert_color = (0, 0, 255)
            elif 2 in self.alert_zones:  # High security
                alert_color = (0, 100, 255)
            
            # Alert overlay
            overlay = np.full_like(frame, alert_color, dtype=np.uint8)
            frame = cv2.addWeighted(frame, 1 - pulse_intensity, overlay, pulse_intensity, 0)
            
            # Alert text with animation
            alert_messages = [
                " SECURITY BREACH DETECTED ",
                f"ZONE(S): {', '.join(map(str, self.alert_zones))}",
                f"TIME: {elapsed:.1f}s"
            ]
            
            for i, message in enumerate(alert_messages):
                text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = 100 + i * 50
                
                # Text shadow
                cv2.putText(frame, message, (text_x + 3, text_y + 3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
                # Main text
                cv2.putText(frame, message, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        else:
            self.alert_active = False
            self.alert_zones.clear()
        
        return frame
    
    def process_enhanced_detections(self, frame: np.ndarray, results) -> np.ndarray:
        """Process YOLO detections with enhanced tracking and analysis"""
        current_objects = {}
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract detection data
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Only process target classes with sufficient confidence
                    if class_id in self.target_classes and confidence > 0.3:
                        object_class = self.target_classes[class_id]
                        centroid = ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2)
                        
                        # Update statistics
                        self.stats['total_detections'] += 1
                        self.stats['class_detections'][object_class] += 1
                        
                        # Update heatmap
                        self.update_heatmap(centroid)
                        
                        # Object tracking for trails
                        obj_key = f"{object_class}_{int(centroid[0]//50)}_{int(centroid[1]//50)}"
                        if obj_key not in self.object_trails:
                            self.object_trails[obj_key] = deque(maxlen=self.trail_length)
                        self.object_trails[obj_key].append(centroid)
                        
                        # Enhanced bounding box visualization
                        box_color = (0, 255, 0)  # Default green
                        box_thickness = 2
                        
                        # Check for intrusions
                        intrusion_zone = None
                        for zone_id, zone in self.zones.items():
                            if (self.zone_enabled[zone_id] and 
                                self.point_in_polygon(centroid, zone)):
                                intrusion_zone = zone_id
                                box_color = self.zone_colors[zone_id]
                                box_thickness = 4
                                self.trigger_alert(zone_id, object_class, centroid, confidence)
                                break
                        
                        # Draw enhanced bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                    box_color, box_thickness)
                        
                        # Object label with enhanced information
                        label_parts = [object_class]
                        if self.show_confidence:
                            label_parts.append(f"{confidence:.2f}")
                        if intrusion_zone:
                            label_parts.append(f"Z{intrusion_zone}")
                        
                        label = " | ".join(label_parts)
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        # Label background
                        cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10), 
                                    (int(x1) + label_size[0] + 10, int(y1)), 
                                    box_color, -1)
                        
                        # Label text
                        cv2.putText(frame, label, (int(x1) + 5, int(y1) - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Enhanced centroid visualization
                        if intrusion_zone:
                            # Pulsing red circle for intrusions
                            pulse_radius = 8 + int(4 * math.sin(time.time() * 20))
                            cv2.circle(frame, centroid, pulse_radius, (0, 0, 255), 3)
                            cv2.circle(frame, centroid, 3, (255, 255, 255), -1)
                        else:
                            # Regular green circle
                            cv2.circle(frame, centroid, 5, (0, 255, 0), -1)
                            cv2.circle(frame, centroid, 8, (0, 255, 0), 2)
        
        # Clean up old trails
        current_time = time.time()
        keys_to_remove = []
        for key, trail in self.object_trails.items():
            if len(trail) == 0 or current_time - getattr(trail, 'last_update', current_time) > 5:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.object_trails[key]
        
        return frame
    
    def handle_enhanced_keyboard_input(self, key: int) -> bool:
        """Handle enhanced keyboard input with more controls"""
        if key == ord('1'):
            self.selected_zone = 1
            print(f"Zone 1 selected: {self.zone_names[1]}")
        elif key == ord('2'):
            self.selected_zone = 2
            print(f"Zone 2 selected: {self.zone_names[2]}")
        elif key == ord('3'):
            self.selected_zone = 3
            print(f"Zone 3 selected: {self.zone_names[3]}")
        elif key == ord('4'):
            self.selected_zone = 4
            print(f"Zone 4 selected: {self.zone_names[4]}")
        elif key == ord('+') or key == ord('='):
            self.resize_zone(self.selected_zone, True)
        elif key == ord('-') or key == ord('_'):
            self.resize_zone(self.selected_zone, False)
        elif key == ord('r'):
            self.rotate_zone(self.selected_zone, math.pi / 8)  # 22.5 degrees
        elif key == ord('R'):  # Shift+R - Reset all zones
            self.reset_all_zones()
        elif key == ord('t'):
            self.toggle_zone(self.selected_zone)
        elif key == ord('h'):
            self.show_heatmap = not self.show_heatmap
            print(f"Heatmap: {'ON' if self.show_heatmap else 'OFF'}")
        elif key == ord('a'):
            self.show_analytics = not self.show_analytics
            print(f"Analytics panel: {'ON' if self.show_analytics else 'OFF'}")
        elif key == ord('v'):
            self.show_trails = not self.show_trails
            print(f"Object trails: {'ON' if self.show_trails else 'OFF'}")
        elif key == ord('c'):
            self.show_confidence = not self.show_confidence
            print(f"Confidence display: {'ON' if self.show_confidence else 'OFF'}")
        elif key == ord('n'):
            self.night_mode = not self.night_mode
            print(f"Night mode: {'ON' if self.night_mode else 'OFF'}")
        elif key == ord('s'):
            self.save_enhanced_configuration()
        elif key == ord('l'):
            self.load_configuration()
        elif key == ord(' '):  # Space bar
            self.toggle_recording()
        elif key == 27:  # ESC key
            self.emergency_stop()
        elif key == ord('q'):
            return False
        
        return True
    
    def toggle_zone(self, zone_id: int):
        """Toggle zone activation"""
        if zone_id in self.zone_enabled:
            self.zone_enabled[zone_id] = not self.zone_enabled[zone_id]
            status = "ENABLED" if self.zone_enabled[zone_id] else "DISABLED"
            print(f"Zone {zone_id} ({self.zone_names[zone_id]}): {status}")
    
    def reset_all_zones(self):
        """Reset all zones to default positions"""
        self.zones = {
            1: self._create_pentagon_zone(0.15, 0.15, 0.35, 0.35),
            2: self._create_hexagon_zone(0.65, 0.65, 0.85, 0.85),
            3: self._create_octagon_zone(0.4, 0.1, 0.6, 0.3),
            4: self._create_triangle_zone(0.1, 0.6, 0.3, 0.9)
        }
        print("All zones reset to default positions")
    
    def save_enhanced_configuration(self):
        """Save enhanced configuration with all settings"""
        config = {
            'zones': {str(k): v.tolist() for k, v in self.zones.items()},
            'zone_enabled': self.zone_enabled,
            'zone_sensitivity': self.zone_sensitivity,
            'zone_names': self.zone_names,
            'settings': {
                'show_heatmap': self.show_heatmap,
                'show_analytics': self.show_analytics,
                'show_trails': self.show_trails,
                'show_confidence': self.show_confidence,
                'night_mode': self.night_mode,
                'trail_length': self.trail_length,
                'alert_duration': self.alert_duration
            },
            'statistics': self.stats,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('advanced_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("Enhanced configuration saved to advanced_config.json")
    
    def load_configuration(self):
        """Load saved configuration"""
        try:
            with open('advanced_config.json', 'r') as f:
                config = json.load(f)
            
            # Load zones
            self.zones = {int(k): np.array(v, dtype=np.int32) 
                         for k, v in config['zones'].items()}
            
            # Load settings
            if 'zone_enabled' in config:
                self.zone_enabled = {int(k): v for k, v in config['zone_enabled'].items()}
            if 'zone_sensitivity' in config:
                self.zone_sensitivity = {int(k): v for k, v in config['zone_sensitivity'].items()}
            if 'settings' in config:
                settings = config['settings']
                self.show_heatmap = settings.get('show_heatmap', False)
                self.show_analytics = settings.get('show_analytics', False)
                self.show_trails = settings.get('show_trails', True)
                self.show_confidence = settings.get('show_confidence', True)
                self.night_mode = settings.get('night_mode', False)
                self.trail_length = settings.get('trail_length', 30)
                self.alert_duration = settings.get('alert_duration', 5.0)
            
            print("Configuration loaded successfully")
            
        except FileNotFoundError:
            print("No saved configuration found")
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    def toggle_recording(self):
        """Toggle video recording"""
        if not self.recording:
            # Start recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"intrusion_recording_{timestamp}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, 
                                              (self.frame_width, self.frame_height))
            self.recording = True
            print(f"Recording started: {filename}")
        else:
            # Stop recording
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.recording = False
            print("Recording stopped")
    
    def emergency_stop(self):
        """Emergency stop all alerts and sounds"""
        self.alert_active = False
        self.alert_zones.clear()
        pygame.mixer.stop()
        print(" EMERGENCY STOP - All alerts cleared")
    
    def update_fps(self):
        """Enhanced FPS calculation"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def run(self):
        """Main enhanced detection loop"""
        print(" Starting Advanced Intrusion Detection System...")
        print("System ready for monitoring...")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Run YOLO detection
                results = self.model(frame, verbose=False)
                
                # Process detections with enhanced features
                frame = self.process_enhanced_detections(frame, results)
                
                # Apply all visualization layers
                frame = self.draw_object_trails(frame)
                frame = self.draw_enhanced_zones(frame)
                frame = self.draw_heatmap(frame)
                frame = self.draw_analytics_panel(frame)
                frame = self.draw_enhanced_alert_overlay(frame)
                frame = self.draw_enhanced_ui(frame)
                
                # Record if active
                if self.recording and self.video_writer:
                    self.video_writer.write(frame)
                
                # Update performance metrics
                self.update_fps()
                
                # Display frame
                cv2.imshow('Advanced Intrusion Detection System', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_enhanced_keyboard_input(key):
                    break
                
        except KeyboardInterrupt:
            print("\nSystem interrupted by user")
        except Exception as e:
            print(f"System error: {e}")
            logging.error(f"System error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Enhanced cleanup"""
        if self.recording and self.video_writer:
            self.video_writer.release()
        
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        
        # Save final statistics
        self.save_enhanced_configuration()
        
        print("Advanced Intrusion Detection System stopped")
        print(f"Session Summary:")
        print(f"  Total Detections: {self.stats['total_detections']}")
        print(f"  Total Intrusions: {self.stats['total_intrusions']}")
        print(f"  Session Duration: {time.time() - self.session_start:.1f}s")

def main():
    """Main function with enhanced error handling"""
    try:
        print(" Initializing Advanced Intrusion Detection System...")
        
        # Check for GPU acceleration
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Initialize system
        system = AdvancedIntrusionDetectionSystem()
        
        # Run the system
        system.run()
        
    except KeyboardInterrupt:
        print("\n System interrupted by user")
    except Exception as e:
        print(f" Critical error: {e}")
        logging.error(f"Critical error: {e}")

if __name__ == "__main__":
    main()