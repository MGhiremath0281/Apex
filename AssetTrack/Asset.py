import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import threading

class AlertStatus(Enum):
    NORMAL = "NORMAL"
    MOVED = "MOVED"

@dataclass
class TrackedObject:
    """Represents a tracked object with movement monitoring"""
    id: str
    class_name: str
    current_position: Tuple[int, int]
    initial_position: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    confidence: float
    last_seen: float
    frames_since_detection: int
    has_moved: bool
    alert_generated: bool
    distance_moved: float
    movement_threshold: float # Individual threshold for this object (now used for reference frame padding)
    reference_frame_bbox: Optional[Tuple[int, int, int, int]] = None # New: Fixed reference area for the object

class MovementMonitor:
    """Real-time object movement monitoring system"""
    
    def __init__(self, model_path: str = "yolov8x.pt", reference_frame_padding_percent: float = 20.0):
        # Initialize YOLO model
        self.model = YOLO(model_path)
        
        # Movement monitoring parameters
        # This threshold now defines the padding percentage for the reference frame around the initial bbox
        self.reference_frame_padding_percent = reference_frame_padding_percent
        self.max_disappeared_frames = 30  # frames before considering object lost
        self.max_distance_for_matching = 100  # maximum distance for object matching
        
        # Tracking data structures
        self.tracked_objects: Dict[str, TrackedObject] = {}
        self.next_object_id = 1
        self.frame_count = 0
        
        # Alert system
        self.active_alerts: List[str] = []
        self.alert_display_duration = 5.0  # seconds
        self.alert_timestamps: Dict[str, float] = {}
        
        # Colors for different object classes
        self.class_colors = self._generate_class_colors()
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Minimum frames to establish initial position (and thus reference frame)
        self.min_frames_for_initial_position = 10
    
    def _generate_class_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Generate colors for different object classes"""
        colors = {}
        self.color_index = 0 # Initialize color_index here
        return colors
    
    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for a specific class"""
        if class_name not in self.class_colors:
            common_colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
                (0, 128, 255), (255, 128, 128), (128, 255, 0), (255, 0, 128)
            ]
            self.class_colors[class_name] = common_colors[self.color_index % len(common_colors)]
            self.color_index += 1
        
        return self.class_colors[class_name]
    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _get_center_point(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def _create_padded_bbox(self, bbox: Tuple[int, int, int, int], padding_percent: float) -> Tuple[int, int, int, int]:
        """
        Creates a padded bounding box from an original bbox.
        Padding is applied as a percentage of the width/height.
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        pad_x = int(width * padding_percent / 100.0)
        pad_y = int(height * padding_percent / 100.0)

        # Ensure padding doesn't push coordinates outside image boundaries (assume 0,0 top-left)
        # We don't have image dimensions here, so will just clip to 0 for min
        new_x1 = max(0, x1 - pad_x)
        new_y1 = max(0, y1 - pad_y)
        new_x2 = x2 + pad_x
        new_y2 = y2 + pad_y
        
        return (new_x1, new_y1, new_x2, new_y2)

    def _is_point_inside_bbox(self, point: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> bool:
        """Checks if a point (x, y) is inside a bounding box (x1, y1, x2, y2)"""
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def _find_best_match(self, new_detections: List, existing_objects: Dict[str, TrackedObject]) -> Dict[str, int]:
        """Find best matches between new detections and existing objects"""
        matches = {}
        used_detections = set()
        
        for obj_id, obj in existing_objects.items():
            best_match_idx = -1
            best_distance = float('inf')
            
            for idx, detection in enumerate(new_detections):
                if idx in used_detections:
                    continue
                
                bbox = detection.xyxy[0].cpu().numpy()
                class_id = int(detection.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                if class_name != obj.class_name:
                    continue
                
                x1, y1, x2, y2 = map(int, bbox)
                center = self._get_center_point((x1, y1, x2, y2))
                distance = self._calculate_distance(obj.current_position, center)
                
                if distance < self.max_distance_for_matching and distance < best_distance:
                    best_distance = distance
                    best_match_idx = idx
            
            if best_match_idx != -1:
                matches[obj_id] = best_match_idx
                used_detections.add(best_match_idx)
        
        return matches
    
    def _update_object_tracking(self, detections: List) -> None:
        """Update object tracking with new detections and establish reference frames."""
        current_time = time.time()
        
        if not detections:
            # No detections, increment frames_since_detection for all objects
            for obj in self.tracked_objects.values():
                obj.frames_since_detection += 1
            return
        
        # Find matches between detections and existing objects
        matches = self._find_best_match(detections, self.tracked_objects)
        
        # Update matched objects
        updated_objects_in_current_frame = set()
        for obj_id, detection_idx in matches.items():
            detection = detections[detection_idx]
            bbox = detection.xyxy[0].cpu().numpy()
            confidence = detection.conf[0].cpu().numpy()
            
            x1, y1, x2, y2 = map(int, bbox)
            center = self._get_center_point((x1, y1, x2, y2))
            
            obj = self.tracked_objects[obj_id]
            obj.current_position = center
            obj.bbox = (x1, y1, x2, y2)
            obj.confidence = confidence
            obj.last_seen = current_time
            obj.frames_since_detection = 0
            
            # Recalculate distance moved from initial position (useful for display)
            obj.distance_moved = self._calculate_distance(obj.initial_position, center)

            # Establish reference frame if not already set and min_frames met
            if obj.reference_frame_bbox is None and obj.frames_since_detection >= self.min_frames_for_initial_position:
                obj.reference_frame_bbox = self._create_padded_bbox(obj.bbox, obj.movement_threshold)
                print(f"Established reference frame for {obj.id}: {obj.reference_frame_bbox}")
            
            updated_objects_in_current_frame.add(obj_id)
        
        # Create new objects for unmatched detections
        used_indices = set(matches.values())
        for idx, detection in enumerate(detections):
            if idx not in used_indices:
                bbox = detection.xyxy[0].cpu().numpy()
                confidence = detection.conf[0].cpu().numpy()
                class_id = int(detection.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                x1, y1, x2, y2 = map(int, bbox)
                center = self._get_center_point((x1, y1, x2, y2))
                
                obj_id = f"{class_name}_{self.next_object_id:03d}"
                self.next_object_id += 1
                
                new_object = TrackedObject(
                    id=obj_id,
                    class_name=class_name,
                    current_position=center,
                    initial_position=center,
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    last_seen=current_time,
                    frames_since_detection=0,
                    has_moved=False,
                    alert_generated=False,
                    distance_moved=0.0,
                    movement_threshold=self.reference_frame_padding_percent, # Use global default for new objects
                    reference_frame_bbox=None # Will be set after min_frames_for_initial_position
                )
                
                self.tracked_objects[obj_id] = new_object
                updated_objects_in_current_frame.add(obj_id)
        
        # Increment frames_since_detection for non-updated objects
        for obj_id, obj in self.tracked_objects.items():
            if obj_id not in updated_objects_in_current_frame:
                obj.frames_since_detection += 1
        
        # Remove objects that haven't been seen for too long
        objects_to_remove = []
        for obj_id, obj in self.tracked_objects.items():
            if obj.frames_since_detection > self.max_disappeared_frames:
                objects_to_remove.append(obj_id)
        
        for obj_id in objects_to_remove:
            del self.tracked_objects[obj_id]
            # Remove any alerts for this object
            if obj_id in self.alert_timestamps:
                del self.alert_timestamps[obj_id]
    
    def _check_movement_alerts(self) -> None:
        """Check for movement alerts if object moves out of its assigned reference frame."""
        current_time = time.time()
        
        for obj_id, obj in self.tracked_objects.items():
            # Only check for movement if a reference frame has been established
            if obj.reference_frame_bbox is not None:
                is_inside_frame = self._is_point_inside_bbox(obj.current_position, obj.reference_frame_bbox)

                if not is_inside_frame:
                    if not obj.alert_generated:
                        obj.has_moved = True
                        obj.alert_generated = True
                        alert_message = (
                            f"{obj.id} ({obj.class_name}) moved OUT of its assigned frame. "
                            f"Current position: {obj.current_position}. "
                            f"Ref Frame: ({obj.reference_frame_bbox[0]},{obj.reference_frame_bbox[1]}) "
                            f"to ({obj.reference_frame_bbox[2]},{obj.reference_frame_bbox[3]})"
                        )
                        if obj_id not in self.active_alerts:
                            self.active_alerts.append(obj_id)
                        self.alert_timestamps[obj_id] = current_time
                        print(f"ALERT: {alert_message}")
                else:
                    # Object is back within its assigned frame
                    if obj.alert_generated:
                        obj.has_moved = False
                        obj.alert_generated = False
                        if obj_id in self.active_alerts:
                            self.active_alerts.remove(obj_id)
                        if obj_id in self.alert_timestamps:
                            del self.alert_timestamps[obj_id]
    
    def _draw_objects(self, frame: np.ndarray) -> None:
        """Draw tracked objects on the frame"""
        for obj_id, obj in self.tracked_objects.items():
            color = self._get_class_color(obj.class_name)
            
            # Draw bounding box
            x1, y1, x2, y2 = obj.bbox
            
            # Use different colors for moved objects (meaning out of reference frame)
            if obj.has_moved:
                box_color = (0, 0, 255)  # Red for objects out of frame
                thickness = 3
            else:
                box_color = color
                thickness = 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
            
            # Draw object ID and class name
            label = f"{obj.id}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
            # Draw confidence
            confidence_text = f"{obj.confidence:.2f}"
            cv2.putText(frame, confidence_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw distance moved from initial position (still useful info)
            if obj.frames_since_detection > self.min_frames_for_initial_position:
                distance_text = f"Moved: {obj.distance_moved:.1f}px"
                cv2.putText(frame, distance_text, (x1, y2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw current position marker
            cv2.circle(frame, obj.current_position, 3, box_color, -1)

    def _draw_reference_frames(self, frame: np.ndarray) -> None:
        """Draws the fixed reference frame for each object."""
        for obj_id, obj in self.tracked_objects.items():
            if obj.reference_frame_bbox is not None:
                x1, y1, x2, y2 = obj.reference_frame_bbox
                
                # Draw the reference frame rectangle
                color = (0, 255, 255) # Cyan
                thickness = 1
                
                # Draw a dashed rectangle for the reference frame (approximate)
                # Drawing segments for dashed line
                dash_length = 10
                gap_length = 5
                
                # Top line
                for i in range(x1, x2, dash_length + gap_length):
                    cv2.line(frame, (i, y1), (min(i + dash_length, x2), y1), color, thickness)
                # Bottom line
                for i in range(x1, x2, dash_length + gap_length):
                    cv2.line(frame, (i, y2), (min(i + dash_length, x2), y2), color, thickness)
                # Left line
                for i in range(y1, y2, dash_length + gap_length):
                    cv2.line(frame, (x1, i), (x1, min(i + dash_length, y2)), color, thickness)
                # Right line
                for i in range(y1, y2, dash_length + gap_length):
                    cv2.line(frame, (x2, i), (x2, min(i + dash_length, y2)), color, thickness)

                # Add a dot at the center of the initial position within the reference frame
                cv2.circle(frame, obj.initial_position, 5, (0, 255, 0), -1) # Green dot, filled
                cv2.putText(frame, "Dot", (obj.initial_position[0] + 10, obj.initial_position[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # Label the reference frame
                frame_label = f"Ref: {obj.movement_threshold:.1f}%"
                cv2.putText(frame, frame_label, (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _draw_alerts(self, frame: np.ndarray) -> None:
        """Draw movement alerts on the frame"""
        if not self.active_alerts:
            return
        
        # Draw alert header
        alert_header = f"MOVEMENT ALERTS ({len(self.active_alerts)})"
        cv2.rectangle(frame, (10, 10), (400, 50), (0, 0, 255), -1)
        cv2.putText(frame, alert_header, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw individual alerts
        y_offset = 70
        for obj_id in self.active_alerts:
            if obj_id in self.tracked_objects:
                obj = self.tracked_objects[obj_id]
                alert_text = f"{obj.id}: OUT OF FRAME!"
                
                # Draw alert background
                text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (10, y_offset - 20), (text_size[0] + 20, y_offset + 5), 
                              (0, 0, 255), -1)
                
                # Draw alert text
                cv2.putText(frame, alert_text, (15, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                y_offset += 35
    
    def _draw_system_info(self, frame: np.ndarray) -> None:
        """Draw system information"""
        info_lines = [
            f"Frame: {self.frame_count}",
            f"Objects: {len(self.tracked_objects)}",
            f"Ref Frame Padding: {self.reference_frame_padding_percent:.1f}%",
            f"Active Alerts: {len(self.active_alerts)}"
        ]
        
        y_offset = frame.shape[0] - 80
        for line in info_lines:
            cv2.putText(frame, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 18
    
    def _draw_instructions(self, frame: np.ndarray) -> None:
        """Draw usage instructions"""
        instructions = [
            "Controls:",
            "Q: Quit",
            "R: Reset objects & re-establish frames",
            "+/-: Adjust global ref frame padding"
        ]
        
        x_offset = frame.shape[1] - 250 # Adjusted for longer text
        y_offset = 30
        
        for instruction in instructions:
            cv2.putText(frame, instruction, (x_offset, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame and return annotated frame"""
        with self.lock:
            self.frame_count += 1
            
            # Run YOLO detection
            results = self.model(frame, conf=0.5, verbose=False)
            
            # Update tracking
            detections = []
            if results and len(results) > 0 and results[0].boxes is not None:
                detections = results[0].boxes
                # Convert to list of individual detections for easier processing
                if len(detections) > 0:
                    detection_list = []
                    for i in range(len(detections)):
                        detection_list.append(detections[i:i+1])
                    detections = detection_list
                else:
                    detections = []
            
            self._update_object_tracking(detections)
            
            # Check for movement alerts
            self._check_movement_alerts()
            
            # Create annotated frame
            annotated_frame = frame.copy()
            
            # Draw all visualizations
            self._draw_reference_frames(annotated_frame) # Draw reference frames first
            self._draw_objects(annotated_frame)
            self._draw_alerts(annotated_frame)
            self._draw_system_info(annotated_frame)
            self._draw_instructions(annotated_frame)
            
            return annotated_frame
    
    def reset_all_objects(self):
        """Reset all tracked objects to their current positions and re-establish reference frames."""
        with self.lock:
            objects_to_reset = list(self.tracked_objects.values()) # Create a copy to avoid RuntimeError during iteration
            for obj in objects_to_reset:
                obj.initial_position = obj.current_position
                obj.distance_moved = 0.0
                obj.has_moved = False
                obj.alert_generated = False
                obj.frames_since_detection = 0 # Reset frames since detection for re-establishment
                obj.movement_threshold = self.reference_frame_padding_percent # Reset individual threshold to global
                
                # Re-establish reference frame based on current bbox and global padding
                # This will be re-evaluated after min_frames_for_initial_position
                obj.reference_frame_bbox = self._create_padded_bbox(obj.bbox, self.reference_frame_padding_percent) 
            
            # Clear all alerts
            self.active_alerts.clear()
            self.alert_timestamps.clear()
            
            print("All objects reset to current positions and reference frames re-established.")
    
    def adjust_threshold(self, delta: float):
        """Adjust reference frame padding percentage (global default)"""
        with self.lock:
            self.reference_frame_padding_percent = max(0.0, min(100.0, self.reference_frame_padding_percent + delta))
            print(f"Global reference frame padding adjusted to {self.reference_frame_padding_percent:.1f}%")
    
    def run_video_stream(self, source: int = 0):
        """Run the movement monitoring system on a video stream"""
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("Starting Movement Monitoring System (Fixed Reference Frame Mode)...")
        print("Objects will have a fixed reference frame established after initial position is stable.")
        print(f"Initial Global Reference Frame Padding: {self.reference_frame_padding_percent:.1f}%")
        print("Controls: Q=Quit, R=Reset, +/-=Adjust GLOBAL reference frame padding")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Movement Monitoring System', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_all_objects()
                elif key == ord('=') or key == ord('+'):
                    self.adjust_threshold(5.0) # Adjust by 5%
                elif key == ord('-'):
                    self.adjust_threshold(-5.0) # Adjust by 5%
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Movement monitoring system stopped")

def main():
    """Main function to run the movement monitoring system"""
    # Initialize the monitor with custom initial global reference frame padding
    monitor = MovementMonitor(reference_frame_padding_percent=20.0) # 20% padding around initial bbox
    
    # Run the monitoring system
    # Use 0 for webcam, or provide path to video file
    monitor.run_video_stream(source=0)

if __name__ == "__main__":
    main()