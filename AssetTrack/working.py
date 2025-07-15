import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict
import math

class ObjectTracker:
    """Enhanced object tracker with individual thresholds and alert system"""
    
    def __init__(self):
        self.objects = {}  # Store tracked objects with their history
        self.next_id = 1
        self.max_distance = 50  # Maximum distance for object association
        self.position_history_size = 10  # Number of positions to keep for smoothing
        self.frame_padding = 20  # Padding around object for reference frame
        
    def update(self, detections):
        """Update tracked objects with new detections"""
        current_objects = {}
        
        # Associate detections with existing objects
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Find closest existing object
            best_match = None
            min_distance = float('inf')
            
            for obj_id, obj_data in self.objects.items():
                if obj_data['class'] == cls:
                    last_pos = obj_data['positions'][-1]
                    distance = math.sqrt((center_x - last_pos[0])**2 + (center_y - last_pos[1])**2)
                    
                    if distance < min_distance and distance < self.max_distance:
                        min_distance = distance
                        best_match = obj_id
            
            if best_match:
                # Update existing object
                obj_data = self.objects[best_match]
                obj_data['positions'].append((center_x, center_y))
                obj_data['bbox'] = (x1, y1, x2, y2)
                obj_data['confidence'] = conf
                obj_data['last_seen'] = time.time()
                
                # Keep only recent positions
                if len(obj_data['positions']) > self.position_history_size:
                    obj_data['positions'].pop(0)
                    
                current_objects[best_match] = obj_data
            else:
                # Create new object
                obj_id = self.next_id
                self.next_id += 1
                
                # Calculate reference frame based on initial detection
                frame_x1 = max(0, x1 - self.frame_padding)
                frame_y1 = max(0, y1 - self.frame_padding)
                frame_x2 = x2 + self.frame_padding
                frame_y2 = y2 + self.frame_padding
                
                current_objects[obj_id] = {
                    'id': obj_id,
                    'class': cls,
                    'positions': [(center_x, center_y)],
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'threshold': 30,  # Default movement threshold
                    'initial_position': (center_x, center_y),
                    'initial_bbox': (x1, y1, x2, y2),
                    'reference_frame': (frame_x1, frame_y1, frame_x2, frame_y2),
                    'reference_dot': (center_x, center_y),
                    'last_seen': time.time(),
                    'alert_active': False,
                    'frame_alert_active': False,
                    'alert_time': 0
                }
        
        # Update objects dictionary
        self.objects = current_objects
        
        return self.objects

class AlertSystem:
    """Manages alerts and visual notifications"""
    
    def __init__(self):
        self.active_alerts = {}
        self.alert_duration = 3.0  # Alert display duration in seconds
        
    def check_alerts(self, tracked_objects):
        """Check for threshold violations and frame exits, manage alerts"""
        current_time = time.time()
        
        for obj_id, obj_data in tracked_objects.items():
            if len(obj_data['positions']) < 2:
                continue
                
            # Calculate distance from initial position
            current_pos = obj_data['positions'][-1]
            initial_pos = obj_data['initial_position']
            distance = math.sqrt((current_pos[0] - initial_pos[0])**2 + 
                               (current_pos[1] - initial_pos[1])**2)
            
            # Check threshold violation
            threshold_violated = distance > obj_data['threshold']
            
            # Check if object is outside reference frame
            ref_frame = obj_data['reference_frame']
            frame_x1, frame_y1, frame_x2, frame_y2 = ref_frame
            frame_violated = not (frame_x1 <= current_pos[0] <= frame_x2 and 
                                frame_y1 <= current_pos[1] <= frame_y2)
            
            class_name = self.get_class_name(obj_data['class'])
            
            # Handle threshold alerts
            if threshold_violated:
                if not obj_data['alert_active']:
                    obj_data['alert_active'] = True
                    obj_data['alert_time'] = current_time
                    
                    print(f"🚨 MOVEMENT ALERT: {class_name} (ID: {obj_id}) moved {distance:.1f} pixels "
                          f"beyond threshold of {obj_data['threshold']} pixels!")
                    
                    self.active_alerts[f"threshold_{obj_id}"] = {
                        'type': 'threshold',
                        'start_time': current_time,
                        'object_data': obj_data,
                        'distance': distance
                    }
            else:
                if obj_data['alert_active']:
                    obj_data['alert_active'] = False
                    if f"threshold_{obj_id}" in self.active_alerts:
                        del self.active_alerts[f"threshold_{obj_id}"]
            
            # Handle frame exit alerts
            if frame_violated:
                if not obj_data['frame_alert_active']:
                    obj_data['frame_alert_active'] = True
                    
                    print(f"🚨 FRAME EXIT ALERT: {class_name} (ID: {obj_id}) has left its reference frame!")
                    
                    self.active_alerts[f"frame_{obj_id}"] = {
                        'type': 'frame_exit',
                        'start_time': current_time,
                        'object_data': obj_data,
                        'distance': distance
                    }
            else:
                if obj_data['frame_alert_active']:
                    obj_data['frame_alert_active'] = False
                    if f"frame_{obj_id}" in self.active_alerts:
                        del self.active_alerts[f"frame_{obj_id}"]
        
        # Remove expired alerts
        expired_alerts = []
        for alert_id, alert_data in self.active_alerts.items():
            if current_time - alert_data['start_time'] > self.alert_duration:
                expired_alerts.append(alert_id)
        
        for alert_id in expired_alerts:
            del self.active_alerts[alert_id]
    
    def get_class_name(self, class_id):
        """Get human-readable class name"""
        # COCO class names
        class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        if 0 <= class_id < len(class_names):
            return class_names[int(class_id)]
        return f"Object_{int(class_id)}"

class RealTimeMonitoringSystem:
    """Main monitoring system class"""
    
    def __init__(self, model_path='yolov8x.pt', confidence_threshold=0.5):
        print("🔄 Initializing Real-Time Object Monitoring System...")
        
        # Initialize YOLO model
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Initialize tracker and alert system
        self.tracker = ObjectTracker()
        self.alert_system = AlertSystem()
        
        # UI state
        self.selected_object = None
        self.show_help = True
        self.threshold_adjustment = 5  # Pixels to adjust per key press
        
        print("✅ System initialized successfully!")
        print("📹 Starting video capture...")
    
    def draw_ui(self, frame, tracked_objects):
        """Draw the user interface on the frame"""
        height, width = frame.shape[:2]
        
        # Draw help text
        if self.show_help:
            help_text = [
                "Controls:",
                "H - Toggle help",
                "Click object to select",
                "+ - Increase threshold",
                "- - Decrease threshold",
                "R - Reset thresholds",
                "ESC - Exit"
            ]
            
            for i, text in enumerate(help_text):
                cv2.putText(frame, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # First, draw all reference frames and dots (so they appear behind objects)
        for obj_id, obj_data in tracked_objects.items():
            # Draw reference frame (always visible)
            ref_frame = obj_data['reference_frame']
            frame_x1, frame_y1, frame_x2, frame_y2 = ref_frame
            
            # Color based on frame violation
            if obj_data['frame_alert_active']:
                frame_color = (0, 0, 255)  # Red if object is outside frame
            else:
                frame_color = (128, 128, 128)  # Gray for normal frame
            
            # Draw reference frame rectangle
            cv2.rectangle(frame, (int(frame_x1), int(frame_y1)), 
                         (int(frame_x2), int(frame_y2)), frame_color, 2)
            
            # Draw reference dot at initial position
            dot_pos = obj_data['reference_dot']
            cv2.circle(frame, (int(dot_pos[0]), int(dot_pos[1])), 5, (255, 255, 0), -1)  # Yellow dot
            cv2.circle(frame, (int(dot_pos[0]), int(dot_pos[1])), 5, (0, 0, 0), 2)  # Black border
            
            # Draw threshold circle around reference dot
            cv2.circle(frame, (int(dot_pos[0]), int(dot_pos[1])), 
                      obj_data['threshold'], (100, 100, 100), 1)
            
            # Add frame label
            class_name = self.alert_system.get_class_name(obj_data['class'])
            frame_label = f"Frame #{obj_id} ({class_name})"
            cv2.putText(frame, frame_label, (int(frame_x1), int(frame_y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, frame_color, 1)
        
        # Draw object information and current positions
        y_offset = 30
        for obj_id, obj_data in tracked_objects.items():
            class_name = self.alert_system.get_class_name(obj_data['class'])
            
            # Calculate distance from initial position
            current_pos = obj_data['positions'][-1]
            initial_pos = obj_data['initial_position']
            distance = math.sqrt((current_pos[0] - initial_pos[0])**2 + 
                               (current_pos[1] - initial_pos[1])**2)
            
            # Determine color and status based on alerts
            if obj_data['frame_alert_active']:
                color = (0, 0, 255)  # Red for frame exit
                status = "FRAME EXIT!"
            elif obj_data['alert_active']:
                color = (0, 100, 255)  # Orange for threshold violation
                status = "THRESHOLD!"
            else:
                color = (0, 255, 0)  # Green for normal
                status = "OK"
            
            # Highlight selected object
            if self.selected_object == obj_id:
                color = (0, 255, 255)  # Yellow for selected
                status = "SELECTED"
            
            # Draw current bounding box
            x1, y1, x2, y2 = obj_data['bbox']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw object label
            label = f"{class_name} #{obj_id} ({status})"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw line from reference dot to current position
            cv2.line(frame, (int(initial_pos[0]), int(initial_pos[1])), 
                    (int(current_pos[0]), int(current_pos[1])), color, 2)
            
            # Check if object is in frame
            ref_frame = obj_data['reference_frame']
            frame_x1, frame_y1, frame_x2, frame_y2 = ref_frame
            in_frame = (frame_x1 <= current_pos[0] <= frame_x2 and 
                       frame_y1 <= current_pos[1] <= frame_y2)
            
            # Object info panel
            frame_status = "IN FRAME" if in_frame else "OUT OF FRAME"
            info_text = f"ID:{obj_id} | {class_name} | T:{obj_data['threshold']}px | D:{distance:.1f}px | {frame_status}"
            cv2.putText(frame, info_text, (width - 700, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25
        
        # Draw active alerts
        alert_y = height - 120
        for alert_id, alert_data in self.alert_system.active_alerts.items():
            obj_data = alert_data['object_data']
            class_name = self.alert_system.get_class_name(obj_data['class'])
            obj_id = obj_data['id']
            
            if alert_data['type'] == 'threshold':
                distance = alert_data['distance']
                alert_text = f" THRESHOLD ALERT: {class_name} #{obj_id} moved {distance:.1f}px beyond threshold!"
            else:  # frame_exit
                alert_text = f" FRAME EXIT ALERT: {class_name} #{obj_id} has left its reference frame!"
            
            cv2.putText(frame, alert_text, (10, alert_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            alert_y -= 30
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for object selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            tracked_objects = param
            
            # Find clicked object
            for obj_id, obj_data in tracked_objects.items():
                x1, y1, x2, y2 = obj_data['bbox']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected_object = obj_id
                    class_name = self.alert_system.get_class_name(obj_data['class'])
                    print(f" Selected {class_name} #{obj_id}")
                    break
    
    def process_keyboard(self, key, tracked_objects):
        """Process keyboard input for threshold adjustment"""
        if key == ord('h') or key == ord('H'):
            self.show_help = not self.show_help
        
        elif key == ord('r') or key == ord('R'):
            # Reset all thresholds
            for obj_data in tracked_objects.values():
                obj_data['threshold'] = 30
            print(" All thresholds reset to 30 pixels")
        
        elif key == ord('+') or key == ord('='):
            if self.selected_object and self.selected_object in tracked_objects:
                obj_data = tracked_objects[self.selected_object]
                obj_data['threshold'] += self.threshold_adjustment
                class_name = self.alert_system.get_class_name(obj_data['class'])
                print(f" Increased threshold for {class_name} #{self.selected_object} to {obj_data['threshold']}px")
        
        elif key == ord('-') or key == ord('_'):
            if self.selected_object and self.selected_object in tracked_objects:
                obj_data = tracked_objects[self.selected_object]
                obj_data['threshold'] = max(10, obj_data['threshold'] - self.threshold_adjustment)
                class_name = self.alert_system.get_class_name(obj_data['class'])
                print(f" Decreased threshold for {class_name} #{self.selected_object} to {obj_data['threshold']}px")
    
    def run(self, source=0):
        """Main execution loop"""
        # Initialize video capture
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(" Error: Could not open video source")
            return
        
        # Set camera resolution for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Set up window with larger size
        cv2.namedWindow('Real-Time Object Monitoring', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-Time Object Monitoring', 1400, 900)  # Set window size
        
        print(" System running! Press 'H' for help, 'ESC' to exit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(" Error: Could not read frame")
                    break
                
                # Run YOLO detection
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                
                # Extract detections
                detections = []
                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    
                    for box, conf, cls in zip(boxes, confidences, classes):
                        detections.append([box[0], box[1], box[2], box[3], conf, cls])
                
                # Update tracker
                tracked_objects = self.tracker.update(detections)
                
                # Check for alerts
                self.alert_system.check_alerts(tracked_objects)
                
                # Draw UI
                self.draw_ui(frame, tracked_objects)
                
                # Set mouse callback
                cv2.setMouseCallback('Real-Time Object Monitoring', self.mouse_callback, tracked_objects)
                
                # Display frame
                cv2.imshow('Real-Time Object Monitoring', frame)
                
                # Process keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
                elif key != 255:  # Any other key
                    self.process_keyboard(key, tracked_objects)
        
        except KeyboardInterrupt:
            print("\n System stopped by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print(" Cleanup completed")

def main():
    """Main function to run the monitoring system"""
    print(" Real-Time Object Monitoring and Alert System")
    print("=" * 50)
    
    # Initialize and run the system
    monitor = RealTimeMonitoringSystem(
        model_path='yolov8x.pt',  # Will download if not present
        confidence_threshold=0.5
    )
    
    # Run with default camera (0) or change to video file path
    monitor.run(source=0)  # Use source='path/to/video.mp4' for video file

if __name__ == "__main__":
    main()