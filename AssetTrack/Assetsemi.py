import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import math
import os
import datetime

# --- Configuration Constants ---
DEFAULT_MOVEMENT_THRESHOLD = 30  # Default pixels for movement alert
MIN_MOVEMENT_THRESHOLD = 10      # Minimum allowed movement threshold
THRESHOLD_ADJUSTMENT_STEP = 5    # Pixels to adjust threshold per key press
ALERT_DISPLAY_DURATION = 3.0     # Alert display duration in seconds (for UI countdown)
ALERT_DEBOUNCE_FRAMES = 5        # Number of consecutive frames alert condition must persist/clear for
MAX_TRACK_AGE_NORMAL = 60        # Max frames without detection before dropping a track (for general object loss / 'camera range exit')
FRAME_EXIT_REMOVAL_SECONDS = 10.0 # Time in seconds an object must be outside its *reference frame* to be removed
UNSETTLED_OBJECT_REMOVAL_SECONDS = 10.0 # Time in seconds to remove an object if it never settles
HISTORY_SIZE = 30                # Number of positions to keep for smoothing and trajectory drawing
FRAME_PADDING = 20               # Padding around initial object bbox for reference frame
IOU_THRESHOLD_ASSOCIATION = 0.4  # IOU threshold for associating detections to existing tracks
MAX_DISTANCE_ASSOCIATION = 250   # ***Increased for better re-association with fast movement***
SETTLE_PERIOD_SECONDS = 5.0      # Time in seconds for a new/re-entering object to settle

# Classes to EXCLUDE from tracking (use class names from CLASS_NAMES)
# 'person' is now explicitly included as per the new requirement
EXCLUDE_CLASSES = ['bed', 'chair', 'person'] 
# Alternatively, specify classes to ONLY TRACK:
# TRACK_ONLY_CLASSES = ['car', 'dog'] 
# If TRACK_ONLY_CLASSES is not empty, only those will be tracked.
# If TRACK_ONLY_CLASSES is empty, EXCLUDE_CLASSES will be used.
TRACK_ONLY_CLASSES = [] # Leave empty to use EXCLUDE_CLASSES

ALERT_LOG_FILE = "alerts.log"

# COCO class names (for convenience) - make sure this list is correct for your YOLO model
CLASS_NAMES = [
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

def get_class_name(class_id):
    """Get human-readable class name from COCO class_names list."""
    try:
        return CLASS_NAMES[int(class_id)]
    except (IndexError, ValueError):
        return f"Object_{int(class_id)}"

class KalmanFilter:
    """A simple 2D Kalman filter for tracking object centroids."""
    def __init__(self, initial_x, initial_y, dt=1.0):
        # State vector: [x, y, vx, vy] - position and velocity
        self.state = np.array([[initial_x], [initial_y], [0.0], [0.0]])
        
        # Transition matrix (A) - describes how state changes over time
        # x_k = x_{k-1} + vx_{k-1}*dt
        # y_k = y_{k-1} + vy_{k-1}*dt
        # vx_k = vx_{k-1}
        # vy_k = vy_{k-1}
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Observation matrix (H) - relates state to measurement [x, y]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance (Q) - uncertainty in the model (tune this)
        # Higher values mean filter trusts model less, measurements more.
        # ***Increased Q for better adaptability to dynamic movement***
        self.Q = np.diag([0.1, 0.1, 0.5, 0.5]) * 0.1 
        
        # Measurement noise covariance (R) - uncertainty in the measurement (tune this)
        # Higher values mean filter trusts measurements less, model more.
        # ***Decreased R to trust measurements more, improving accuracy***
        self.R = np.diag([3.0, 3.0]) 
        
        # Error covariance matrix (P) - uncertainty in state estimate
        # High initial uncertainty to allow the filter to quickly converge to initial measurements
        self.P = np.diag([100.0, 100.0, 10.0, 10.0])

        # Identity matrix for convenience
        self.I = np.eye(self.state.shape[0])

    def predict(self):
        """Predict the next state."""
        self.state = self.A @ self.state
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.state[0:2].flatten() # Return predicted x, y coordinates

    def update(self, measurement):
        """Update state with new measurement."""
        Z = np.array([[measurement[0]], [measurement[1]]]) # Convert measurement to column vector
        
        Y = Z - (self.H @ self.state)  # Innovation (measurement residual)
        S = self.H @ self.P @ self.H.T + self.R # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman gain
        
        self.state = self.state + (K @ Y)
        self.P = (self.I - (K @ self.H)) @ self.P
        
        return self.state[0:2].flatten() # Return updated x, y coordinates

class ObjectTracker:
    """Enhanced object tracker with individual thresholds, Kalman filter, and class filtering."""
    
    def __init__(self, alert_system_instance): # Pass alert_system instance here
        self.objects = {}  # Store tracked objects with their history
        self.next_id = 1
        self.max_distance = MAX_DISTANCE_ASSOCIATION  # Maximum distance for object association (Euclidean)
        self.position_history_size = HISTORY_SIZE      # Number of positions to keep for smoothing
        self.frame_padding = FRAME_PADDING            # Padding around object for reference frame
        self.max_track_age_normal = MAX_TRACK_AGE_NORMAL # Max frames without detection before dropping a track (for non-frame-exit objects)
        self.frame_exit_removal_seconds = FRAME_EXIT_REMOVAL_SECONDS # Time for object removal after frame exit
        self.unsettled_object_removal_seconds = UNSETTLED_OBJECT_REMOVAL_SECONDS # Time to remove if never settles
        self.iou_threshold_association = IOU_THRESHOLD_ASSOCIATION # IOU for strong association
        self.settle_period_seconds = SETTLE_PERIOD_SECONDS # Time in seconds for new objects to settle
        self.alert_system = alert_system_instance # Store the alert system instance

        # Class filtering logic
        self.track_only_classes_ids = []
        self.exclude_classes_ids = []
        
        if TRACK_ONLY_CLASSES:
            for cls_name in TRACK_ONLY_CLASSES:
                try:
                    self.track_only_classes_ids.append(CLASS_NAMES.index(cls_name))
                except ValueError:
                    print(f"Warning: Class '{cls_name}' in TRACK_ONLY_CLASSES not found in CLASS_NAMES.")
            print(f"Tracking ONLY classes: {[get_class_name(cid) for cid in self.track_only_classes_ids]}")
        elif EXCLUDE_CLASSES:
            for cls_name in EXCLUDE_CLASSES:
                try:
                    self.exclude_classes_ids.append(CLASS_NAMES.index(cls_name))
                except ValueError:
                    print(f"Warning: Class '{cls_name}' in EXCLUDE_CLASSES not found in CLASS_NAMES.")
            print(f"Excluding classes: {[get_class_name(cid) for cid in self.exclude_classes_ids]}")

    def _should_track_class(self, class_id):
        """Determines if a given class_id should be tracked based on configuration."""
        if self.track_only_classes_ids:
            return int(class_id) in self.track_only_classes_ids
        elif self.exclude_classes_ids:
            return int(class_id) not in self.exclude_classes_ids
        return True # Track all if no specific filters are set

    def update(self, detections, frame_shape):
        """Update tracked objects with new detections, using Kalman filter and IOU/distance association."""
        current_active_ids = set()
        current_time = time.time() # Get current time for settle period

        # 1. Filter detections based on class
        filtered_detections = [d for d in detections if self._should_track_class(d[5])]

        # Prepare for association
        unmatched_detections_indices = list(range(len(filtered_detections)))
        unmatched_track_ids = list(self.objects.keys())
        
        # Stores final matches for this frame: {detection_idx: obj_id}
        matches = {} 

        # --- Association Pass 1: Strong IOU Match (Priority 1) ---
        iou_matches = [] # (obj_id, detection_index, iou_score)
        for det_idx in unmatched_detections_indices:
            detection = filtered_detections[det_idx]
            det_x1, det_y1, det_x2, det_y2, _, det_cls = detection
            
            for obj_id in unmatched_track_ids:
                obj_data = self.objects[obj_id]
                # Only consider matching class and an IOU match
                if obj_data['class'] == det_cls: 
                    # Use last known bbox for IOU calculation, as it represents the object's extent
                    track_x1, track_y1, track_x2, track_y2 = obj_data['bbox'] 
                    
                    iou = self._calculate_iou((det_x1, det_y1, det_x2, det_y2), 
                                              (track_x1, track_y1, track_x2, track_y2))
                    
                    if iou > self.iou_threshold_association:
                        iou_matches.append((obj_id, det_idx, iou))
        
        # Sort IOU matches by score (highest first) to resolve conflicts
        iou_matches.sort(key=lambda x: x[2], reverse=True)
        
        for obj_id, det_idx, iou_score in iou_matches:
            if obj_id in unmatched_track_ids and det_idx in unmatched_detections_indices:
                matches[det_idx] = obj_id
                unmatched_track_ids.remove(obj_id)
                unmatched_detections_indices.remove(det_idx)

        # --- Association Pass 2: Kalman Predicted Position + Distance Match (Priority 2) ---
        distance_matches = [] # (obj_id, detection_index, distance)
        for det_idx in unmatched_detections_indices:
            detection = filtered_detections[det_idx]
            det_x1, det_y1, det_x2, det_y2, _, det_cls = detection
            det_center_x = (det_x1 + det_x2) / 2
            det_center_y = (det_y1 + det_y2) / 2
            
            for obj_id in unmatched_track_ids:
                obj_data = self.objects[obj_id]
                if obj_data['class'] == det_cls: # Only match same class
                    # Use predicted position for distance calculation (Kalman filter's best guess)
                    predicted_pos = obj_data['kalman_filter'].state[0:2].flatten() # Use current KF state after prediction
                    
                    distance = math.sqrt((det_center_x - predicted_pos[0])**2 + 
                                         (det_center_y - predicted_pos[1])**2)
                    
                    if distance < self.max_distance:
                        distance_matches.append((obj_id, det_idx, distance))
        
        # Sort distance matches by distance (lowest first)
        distance_matches.sort(key=lambda x: x[2])

        for obj_id, det_idx, distance in distance_matches:
            if obj_id in unmatched_track_ids and det_idx in unmatched_detections_indices:
                matches[det_idx] = obj_id
                unmatched_track_ids.remove(obj_id)
                unmatched_detections_indices.remove(det_idx)
        
        # --- Update Matched Tracks ---
        for det_idx, obj_id in matches.items():
            detection = filtered_detections[det_idx]
            x1, y1, x2, y2, conf, cls = detection
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            obj_data = self.objects[obj_id]
            updated_pos = obj_data['kalman_filter'].update((center_x, center_y)) # Update KF with measurement
            
            obj_data['positions'].append(tuple(updated_pos))
            obj_data['bbox'] = (x1, y1, x2, y2) # Update bbox to the new detection's bbox
            obj_data['confidence'] = conf
            obj_data['last_seen'] = current_time
            obj_data['frames_since_last_seen'] = 0 # Reset age for detected tracks
            obj_data['unmatched_frames_count'] = 0 # Reset unmatched count
            
            # If re-acquired after being lost or after a frame exit alert, force it to re-settle.
            if obj_data['has_settled'] and (obj_data['frame_alert_active'] or obj_data['mark_for_removal']):
                 obj_data['settle_start_time'] = current_time
                 obj_data['has_settled'] = False # Must re-settle
                 class_name = get_class_name(obj_data['class'])
                 print(f"♻️ Re-acquired {class_name} (ID: {obj_id}). Starting re-settle period.")
            
            # Crucially, if detected again, clear any frame exit alert and reset its timer.
            obj_data['mark_for_removal'] = False 
            obj_data['frame_alert_active'] = False # Clear active frame alert
            obj_data['frame_exit_start_time'] = None # Reset frame exit timer

            # Check and update settle status: only set baseline AFTER settle period
            if not obj_data['has_settled'] and (current_time - obj_data['settle_start_time']) >= self.settle_period_seconds:
                obj_data['has_settled'] = True
                # Set initial position and reference frame to current Kalman-filtered position
                obj_data['initial_position'] = tuple(updated_pos)
                obj_data['reference_dot'] = tuple(updated_pos)
                # Recalculate reference frame based on its current settled position/bbox
                x1_current, y1_current, x2_current, y2_current = obj_data['bbox']
                frame_x1 = max(0, int(x1_current - self.frame_padding))
                frame_y1 = max(0, int(y1_current - self.frame_padding))
                frame_x2 = min(frame_shape[1], int(x2_current + self.frame_padding))
                frame_y2 = min(frame_shape[0], int(y2_current + self.frame_padding))
                obj_data['reference_frame'] = (frame_x1, frame_y1, frame_x2, frame_y2)
                class_name = get_class_name(obj_data['class'])
                print(f"✅ {class_name} (ID: {obj_id}) has settled. Movement tracking active.")

            if len(obj_data['positions']) > self.position_history_size:
                obj_data['positions'].popleft()
            
            current_active_ids.add(obj_id)
        
        # --- Handle Unmatched Detections (New Objects) ---
        for det_idx in unmatched_detections_indices:
            detection = filtered_detections[det_idx]
            x1, y1, x2, y2, conf, cls = detection
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            obj_id = self.next_id
            self.next_id += 1
            
            # For new objects, initial_position, reference_frame, reference_dot
            # are temporarily set to their *first detection*. They will be updated
            # to their "settled" position/frame *after* the settle period.
            # IMPORTANT: These temporary values are NOT used for drawing reference frame/dot or alerts until settled.
            frame_x1 = max(0, int(x1 - self.frame_padding))
            frame_y1 = max(0, int(y1 - self.frame_padding))
            frame_x2 = min(frame_shape[1], int(x2 + self.frame_padding))
            frame_y2 = min(frame_shape[0], int(y2 + self.frame_padding))
            
            self.objects[obj_id] = {
                'id': obj_id,
                'class': cls,
                'positions': deque([(center_x, center_y)]), 
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'threshold': DEFAULT_MOVEMENT_THRESHOLD, 
                'initial_position': (center_x, center_y), # Temporary placeholder, will be set on settle
                'initial_bbox': (x1, y1, x2, y2),
                'reference_frame': (frame_x1, frame_y1, frame_x2, frame_y2), # Temporary placeholder, will be set on settle
                'reference_dot': (center_x, center_y), # Temporary placeholder, will be set on settle
                'last_seen': current_time,
                'alert_active': False,
                'frame_alert_active': False,
                'alert_start_time': 0,
                'alert_debounce_counter': 0, 
                'frame_alert_debounce_counter': 0, 
                'frames_since_last_seen': 0, 
                'kalman_filter': KalmanFilter(center_x, center_y), 
                'unmatched_frames_count': 0,
                'settle_start_time': current_time,       # Mark time of creation for settle period
                'has_settled': False,                    # Flag to indicate if settle period is over
                'mark_for_removal': False,               # Flag to explicitly remove object after frame exit alert
                'frame_exit_start_time': None           # Timestamp when object first leaves its frame (for 10s countdown)
            }
            current_active_ids.add(obj_id)
            print(f"➕ New object tracked: {get_class_name(cls)} (ID: {obj_id}). Starting settle period.")
        
        # --- Handle Unmatched Existing Objects (Track Prediction and Aging) ---
        objects_to_remove = []
        for obj_id in list(self.objects.keys()): # Iterate over a copy of keys
            if obj_id not in current_active_ids:
                obj_data = self.objects[obj_id]
                
                # Check for removal of unsettled objects first
                if not obj_data['has_settled'] and \
                   (current_time - obj_data['settle_start_time']) >= self.unsettled_object_removal_seconds:
                    objects_to_remove.append(obj_id)
                    # No specific alert for this, it's just a cleanup of a non-viable track
                    continue # Skip further processing for this object as it's being removed

                obj_data['unmatched_frames_count'] += 1
                
                # Predict position for unmatched tracks using Kalman filter
                predicted_pos = obj_data['kalman_filter'].predict()
                
                # Update bbox based on prediction for visualization purposes (size remains constant)
                last_bbox = obj_data['bbox']
                width = last_bbox[2] - last_bbox[0]
                height = last_bbox[3] - last_bbox[1]
                
                pred_x1 = predicted_pos[0] - width / 2
                pred_y1 = predicted_pos[1] - height / 2
                pred_x2 = predicted_pos[0] + width / 2
                pred_y2 = predicted_pos[1] + height / 2
                
                obj_data['bbox'] = (pred_x1, pred_y1, pred_x2, pred_y2)
                obj_data['positions'].append(tuple(predicted_pos)) # Add predicted position to history
                if len(obj_data['positions']) > self.position_history_size:
                    obj_data['positions'].popleft()

                # Even if unmatched, check for settle period and update baseline if needed
                if not obj_data['has_settled'] and (current_time - obj_data['settle_start_time']) >= self.settle_period_seconds:
                    obj_data['has_settled'] = True
                    # Set initial position and reference frame to current Kalman-filtered position
                    obj_data['initial_position'] = tuple(predicted_pos) # Use predicted pos for re-baselining
                    obj_data['reference_dot'] = tuple(predicted_pos)
                    # Recalculate reference frame based on its current estimated bbox
                    x1_current, y1_current, x2_current, y2_current = obj_data['bbox']
                    frame_x1 = max(0, int(x1_current - self.frame_padding))
                    frame_y1 = max(0, int(y1_current - self.frame_padding))
                    frame_x2 = min(frame_shape[1], int(x2_current + self.frame_padding))
                    frame_y2 = min(frame_shape[0], int(y2_current + self.frame_padding))
                    obj_data['reference_frame'] = (frame_x1, frame_y1, frame_x2, frame_y2)
                    class_name = get_class_name(obj_data['class'])
                    print(f"✅ {class_name} (ID: {obj_id}) has settled (while unmatched). Movement tracking active.")


                # Removal logic: Prioritize 'mark_for_removal' based on 10-sec frame exit.
                # Otherwise, consider normal aging for 'camera range exit'
                if obj_data.get('mark_for_removal', False) and \
                   obj_data['frame_exit_start_time'] is not None and \
                   (current_time - obj_data['frame_exit_start_time']) >= self.frame_exit_removal_seconds:
                    objects_to_remove.append(obj_id)
                    self.alert_system.queue_alert(obj_id, get_class_name(obj_data['class']), "Frame Exit Removal", 
                                                 f"Object left reference frame for {self.frame_exit_removal_seconds}s and was removed.")
                elif not obj_data.get('mark_for_removal', False) and obj_data['unmatched_frames_count'] > self.max_track_age_normal:
                    objects_to_remove.append(obj_id)
                    self.alert_system.queue_alert(obj_id, get_class_name(obj_data['class']), "Camera Range Exit", 
                                                 f"Object lost from camera view for {self.max_track_age_normal} frames.")
        
        for obj_id in objects_to_remove:
            class_name = get_class_name(self.objects[obj_id]['class'])
            removal_reason = "due to age."
            
            # Refine removal message based on specific condition
            if self.objects[obj_id].get('mark_for_removal', False) and self.objects[obj_id].get('frame_exit_start_time') is not None:
                removal_reason = f"after {self.frame_exit_removal_seconds}s frame exit."
            elif not self.objects[obj_id]['has_settled']: # Special reason for unsettled removal
                 removal_reason = f"as unsettled object after {self.unsettled_object_removal_seconds}s."
            else: # Default for general loss/camera range exit
                removal_reason = f"from camera range (not seen for {self.max_track_age_normal} frames)."


            print(f"🗑️ Dropping track for {class_name} (ID: {obj_id}) {removal_reason}")
            
            # Ensure the alert system also cleans up alerts for this ID
            self.alert_system.remove_object_alerts(obj_id) 
            del self.objects[obj_id] # Remove from main dictionary

        return self.objects

    def _calculate_iou(self, boxA, boxB):
        """Calculates Intersection Over Union (IOU) of two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_width = xB - xA
        inter_height = yB - yA

        if inter_width <= 0 or inter_height <= 0:
            return 0.0

        inter_area = inter_width * inter_height
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        iou = inter_area / float(boxA_area + boxB_area - inter_area)
        return iou

class AlertSystem:
    """Manages alerts, visual notifications, and logging."""
    
    def __init__(self, alert_log_file=ALERT_LOG_FILE):
        self.active_alerts = {}
        self.queued_alerts_for_display = deque() # New queue for alerts that should show briefly on removal
        self.alert_duration = ALERT_DISPLAY_DURATION
        self.alert_debounce_frames = ALERT_DEBOUNCE_FRAMES
        self.alert_log_file = alert_log_file
        self.frame_exit_removal_seconds = FRAME_EXIT_REMOVAL_SECONDS # Access constant here
        self._initialize_log_file()
        
    def _initialize_log_file(self):
        """Ensures the alert log file exists and has a header."""
        if not os.path.exists(self.alert_log_file):
            with open(self.alert_log_file, 'w') as f:
                f.write("Timestamp,Object ID,Class Name,Alert Type,Description\n")

    def log_alert(self, obj_id, class_name, alert_type, description):
        """Logs an alert to the specified file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.alert_log_file, 'a') as f:
            f.write(f"{timestamp},{obj_id},{class_name},{alert_type},\"{description}\"\n")

    def queue_alert(self, obj_id, class_name, alert_type, description):
        """Queues an alert for display and logging, typically for removal events."""
        self.log_alert(obj_id, class_name, alert_type, description)
        self.queued_alerts_for_display.append({
            'type': alert_type,
            'object_id': obj_id,
            'class_name': class_name,
            'description': description,
            'start_time': time.time()
        })

    def remove_object_alerts(self, obj_id):
        """Removes all active alerts associated with a specific object ID."""
        keys_to_remove = [key for key in self.active_alerts if str(obj_id) in key]
        for key in keys_to_remove:
            del self.active_alerts[key]
        # Remove from queued alerts too if, for some reason, object is re-acquired quickly
        self.queued_alerts_for_display = deque([
            alert for alert in self.queued_alerts_for_display if alert['object_id'] != obj_id
        ])

    def check_alerts(self, tracked_objects):
        """Check for threshold violations and frame exits, manage alerts with debouncing."""
        current_time = time.time()
        
        # Temporary dictionary to hold active alerts for the current frame
        current_frame_alerts = {} 

        for obj_id, obj_data in tracked_objects.items():
            # Use the most recent position from the deque, which could be a measurement or a prediction
            if not obj_data['positions']: 
                continue
                
            current_pos = obj_data['positions'][-1]
            
            # Important: Only calculate distance for movement alerts if the object has settled
            distance = 0
            if obj_data['has_settled']:
                initial_pos = obj_data['initial_position']
                distance = math.sqrt((current_pos[0] - initial_pos[0])**2 + 
                                     (current_pos[1] - initial_pos[1])**2)
            
            threshold_violated = distance > obj_data['threshold']
            
            frame_violated = False
            # Frame exit check only relevant if object has settled and has an established reference frame
            if obj_data['has_settled']:
                ref_frame = obj_data['reference_frame']
                frame_x1, frame_y1, frame_x2, frame_y2 = ref_frame
                frame_violated = not (frame_x1 <= current_pos[0] <= frame_x2 and 
                                      frame_y1 <= current_pos[1] <= frame_y2)
            
            class_name = get_class_name(obj_data['class'])
            
            # --- Handle Threshold Alerts ---
            # Movement alerts are ONLY active if the object has settled AND threshold is violated
            if threshold_violated and obj_data['has_settled']: 
                obj_data['alert_debounce_counter'] = min(obj_data['alert_debounce_counter'] + 1, self.alert_debounce_frames)
                if obj_data['alert_debounce_counter'] >= self.alert_debounce_frames:
                    if not obj_data['alert_active']:
                        obj_data['alert_active'] = True
                        obj_data['alert_start_time'] = current_time
                        print(f"🚨 MOVEMENT ALERT: {class_name} (ID: {obj_id}) moved {distance:.1f} pixels "
                              f"beyond threshold of {obj_data['threshold']} pixels!")
                        self.log_alert(obj_id, class_name, "Movement", 
                                       f"Moved {distance:.1f}px beyond {obj_data['threshold']}px threshold.")
                    current_frame_alerts[f"threshold_{obj_id}"] = {
                        'type': 'threshold',
                        'start_time': obj_data['alert_start_time'], 
                        'object_data': obj_data,
                        'distance': distance
                    }
            else:
                obj_data['alert_debounce_counter'] = max(obj_data['alert_debounce_counter'] - 1, 0)
                if obj_data['alert_active'] and obj_data['alert_debounce_counter'] == 0:
                    obj_data['alert_active'] = False

            # --- Handle Frame Exit Alerts ---
            # Frame exit alerts are also only active if the object has settled AND frame is violated
            if frame_violated and obj_data['has_settled']:
                obj_data['frame_alert_debounce_counter'] = min(obj_data['frame_alert_debounce_counter'] + 1, self.alert_debounce_frames)
                if obj_data['frame_alert_debounce_counter'] >= self.alert_debounce_frames:
                    if not obj_data['frame_alert_active']:
                        obj_data['frame_alert_active'] = True
                        obj_data['alert_start_time'] = current_time 
                        # Start the frame exit removal timer ONLY when alert first becomes active
                        obj_data['frame_exit_start_time'] = current_time 
                        print(f"🚨 FRAME EXIT ALERT: {class_name} (ID: {obj_id}) has left its reference frame! Starting {self.frame_exit_removal_seconds}s removal timer.")
                        self.log_alert(obj_id, class_name, "Frame Exit", 
                                       f"Left reference frame: ({ref_frame[0]:.0f},{ref_frame[1]:.0f})-({ref_frame[2]:.0f},{ref_frame[3]:.0f}).")
                        obj_data['mark_for_removal'] = True # Signal tracker to consider for removal
                    current_frame_alerts[f"frame_{obj_id}"] = {
                        'type': 'frame_exit',
                        'start_time': obj_data['alert_start_time'],
                        'object_data': obj_data,
                        'distance': distance 
                    }
            else:
                # If object re-enters its frame, clear frame exit alert flags and unmark for removal
                obj_data['frame_alert_debounce_counter'] = max(obj_data['frame_alert_debounce_counter'] - 1, 0)
                if obj_data['frame_alert_active'] and obj_data['frame_alert_debounce_counter'] == 0: # Ensure debounce cleared
                    obj_data['frame_alert_active'] = False
                    obj_data['mark_for_removal'] = False # If it comes back into the frame, unmark for removal
                    obj_data['frame_exit_start_time'] = None # Reset the removal timer
                    print(f"✅ {class_name} (ID: {obj_id}) re-entered frame. Frame exit timer reset.") # Debugging

        # Manage active_alerts based on current_frame_alerts and expiry
        # Add new/persistent alerts from current frame
        for alert_id, alert_data in current_frame_alerts.items():
            self.active_alerts[alert_id] = alert_data
            
        # Remove expired alerts from display
        expired_alerts = []
        for alert_id, alert_data in self.active_alerts.items():
            # Check if object no longer exists in tracked_objects (meaning it was removed by tracker)
            # This handles cases where tracker removed an object before alert display duration finished.
            if alert_data['type'] in ['threshold', 'frame_exit'] and alert_data['object_data']['id'] not in tracked_objects:
                expired_alerts.append(alert_id)
                continue 

            is_threshold_alert_expired = (alert_data['type'] == 'threshold' and 
                                         not tracked_objects.get(alert_data['object_data']['id'], {}).get('alert_active', False) and 
                                         (time.time() - alert_data['start_time'] > self.alert_duration))
            
            is_frame_alert_expired = (alert_data['type'] == 'frame_exit' and 
                                      not tracked_objects.get(alert_data['object_data']['id'], {}).get('frame_alert_active', False) and 
                                      (time.time() - alert_data['start_time'] > self.alert_duration))
            
            # This is for the queued alerts (like 'Camera Range Exit' or 'Frame Exit Removal')
            is_queued_alert_expired = (alert_data['type'] not in ['threshold', 'frame_exit'] and 
                                      (time.time() - alert_data['start_time'] > self.alert_duration))

            if is_threshold_alert_expired or is_frame_alert_expired or is_queued_alert_expired:
                expired_alerts.append(alert_id)
        
        for alert_id in expired_alerts:
            if alert_id in self.active_alerts: 
                del self.active_alerts[alert_id]

        # Add newly queued alerts to active alerts for display
        while self.queued_alerts_for_display:
            alert = self.queued_alerts_for_display.popleft()
            self.active_alerts[f"{alert['type'].replace(' ', '_')}_{alert['object_id']}"] = {
                'type': alert['type'],
                'start_time': alert['start_time'],
                'object_id': alert['object_id'], # Keep object_id directly
                'class_name': alert['class_name'], # Keep class_name directly
                'description': alert['description'] # Store description for display
            }


class RealTimeMonitoringSystem:
    """Main monitoring system class combining detection, tracking, and alerts."""
    
    def __init__(self, model_path='yolov8x.pt', confidence_threshold=0.5):
        print("🔄 Initializing Real-Time Object Monitoring System...")
        
        # Initialize YOLO model
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Initialize alert system FIRST, then pass it to the tracker
        self.alert_system = AlertSystem()
        self.tracker = ObjectTracker(self.alert_system) # Pass the alert system instance here
        
        # UI state
        self.selected_object = None
        self.show_help = True
        
        # Performance metrics
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.detection_time_ms = 0.0
        self.tracking_time_ms = 0.0
        self.threshold_adjustment = THRESHOLD_ADJUSTMENT_STEP # Added this to class for access

        print("✅ System initialized successfully!")
        print("📹 Starting video capture...")
    
    def draw_ui(self, frame, tracked_objects):
        """Draw the user interface on the frame."""
        height, width = frame.shape[:2]
        
        # Draw help text (top-left)
        if self.show_help:
            help_text = [
                "Controls:",
                "H - Toggle help",
                "Click object to select",
                "+/- - Adjust threshold (selected obj)",
                "S - Set new origin (selected obj)",
                "R - Reset all thresholds/origins",
                "ESC - Exit"
            ]
            
            for i, text in enumerate(help_text):
                cv2.putText(frame, text, (10, 30 + i * 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # First, draw all reference frames and dots (so they appear behind objects)
        for obj_id, obj_data in tracked_objects.items():
            # ONLY DRAW REFERENCE FRAME AND DOT IF THE OBJECT HAS SETTLED
            if obj_data['has_settled']:
                ref_frame = obj_data['reference_frame']
                frame_x1, frame_y1, frame_x2, frame_y2 = ref_frame
                
                # Color based on frame violation
                frame_color = (128, 128, 128)  # Default Gray for normal frame
                if obj_data['frame_alert_active']:
                    frame_color = (0, 0, 255)  # Red if object is outside frame
                    # Calculate time remaining for removal if alert is active
                    if obj_data['frame_exit_start_time'] is not None:
                        time_left = self.tracker.frame_exit_removal_seconds - (time.time() - obj_data['frame_exit_start_time'])
                        # Display countdown directly on the frame
                        cv2.putText(frame, f"REMOVING IN: {max(0, time_left):.1f}s", 
                                    (int(frame_x1), int(frame_y2) + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Draw reference frame rectangle
                cv2.rectangle(frame, (int(frame_x1), int(frame_y1)), 
                            (int(frame_x2), int(frame_y2)), frame_color, 2, cv2.LINE_AA)
                
                # Draw reference dot at initial position
                dot_pos = obj_data['reference_dot']
                cv2.circle(frame, (int(dot_pos[0]), int(dot_pos[1])), 5, (255, 255, 0), -1, cv2.LINE_AA)  # Yellow dot
                cv2.circle(frame, (int(dot_pos[0]), int(dot_pos[1])), 5, (0, 0, 0), 2, cv2.LINE_AA)  # Black border
                
                # Draw threshold circle around reference dot
                cv2.circle(frame, (int(dot_pos[0]), int(dot_pos[1])), 
                        obj_data['threshold'], (100, 100, 100), 1, cv2.LINE_AA)
                
                # Add frame label
                class_name = get_class_name(obj_data['class'])
                frame_label = f"Ref #{obj_id} ({class_name})" 
                cv2.putText(frame, frame_label, (int(frame_x1), int(frame_y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, frame_color, 1, cv2.LINE_AA)
        
        # Draw object information, current positions, and trajectories
        y_offset_info = 30 # Y offset for the object info panel (top-right)
        
        for obj_id, obj_data in tracked_objects.items():
            class_name = get_class_name(obj_data['class'])
            
            # Calculate distance from initial position (using the last known position from deque)
            current_pos = obj_data['positions'][-1]
            # Only calculate meaningful distance if the object has settled
            distance = 0
            if obj_data['has_settled']:
                initial_pos = obj_data['initial_position'] 
                distance = math.sqrt((current_pos[0] - initial_pos[0])**2 + 
                                     (current_pos[1] - initial_pos[1])**2)
            
            # Determine color and status based on alerts and settle period
            color = (0, 255, 0)  # Default Green for normal
            status = "OK"

            if not obj_data['has_settled']:
                time_to_settle = max(0, self.tracker.settle_period_seconds - (time.time() - obj_data['settle_start_time']))
                time_to_unsettled_remove = max(0, self.tracker.unsettled_object_removal_seconds - (time.time() - obj_data['settle_start_time']))
                
                color = (255, 165, 0) # Orange for settling objects
                status = f"SETTLING ({time_to_settle:.1f}s) | REMOVE IN ({time_to_unsettled_remove:.1f}s)"
            elif obj_data['frame_alert_active']:
                color = (0, 0, 255)  # Red for frame exit
                status = "FRAME EXIT!"
            elif obj_data['alert_active']:
                color = (0, 100, 255)  # Orange for threshold violation
                status = "THRESHOLD!"
            
            # Highlight selected object
            if self.selected_object == obj_id:
                color = (0, 255, 255)  # Yellow for selected
                status = "SELECTED"
            
            # Draw current bounding box (using the bbox property, which is updated even for predicted frames)
            x1, y1, x2, y2 = obj_data['bbox']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)
            
            # Draw object label and confidence
            label = f"{class_name} #{obj_id} ({status}) C:{obj_data['confidence']:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            
            # Draw trajectory path (from history)
            if len(obj_data['positions']) > 1:
                for i in range(1, len(obj_data['positions'])):
                    pt1 = (int(obj_data['positions'][i-1][0]), int(obj_data['positions'][i-1][1]))
                    pt2 = (int(obj_data['positions'][i][0]), int(obj_data['positions'][i][1]))
                    cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA) # Thin line for trajectory

            # Draw line from reference dot to current position (only if settled)
            if obj_data['has_settled']:
                initial_pos = obj_data['initial_position']
                cv2.line(frame, (int(initial_pos[0]), int(initial_pos[1])), 
                                (int(current_pos[0]), int(current_pos[1])), color, 2, cv2.LINE_AA)
            
            # Check if object is in frame (for info display)
            # This check uses the current reference_frame which is only valid *after* settling
            in_frame_status_display = ""
            if obj_data['has_settled']: # Only show "IN FRAME" / "OUT OF FRAME" if settled
                ref_frame = obj_data['reference_frame']
                frame_x1, frame_y1, frame_x2, frame_y2 = ref_frame
                in_frame_status = "IN FRAME" if (frame_x1 <= current_pos[0] <= frame_x2 and 
                                                frame_y1 <= current_pos[1] <= frame_y2) else "OUT OF FRAME"
                in_frame_status_display = f"| {in_frame_status}"
            
            # Object info panel (right side)
            info_text = f"ID:{obj_id} | {class_name} | T:{obj_data['threshold']}px | D:{distance:.1f}px {in_frame_status_display}"
            # Adjust X coordinate based on text length or fixed position for right alignment
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(frame, info_text, (width - text_size[0] - 10, y_offset_info), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            y_offset_info += 25
        
        # Draw active alerts (bottom left)
        alert_y = height - 60 # Start drawing alerts from bottom up
        active_alert_count = 0
        
        # Create a sorted list of alerts for consistent display order
        # Prioritize "Frame Exit" and "Camera Range Exit" alerts to be higher
        sorted_alerts = sorted(self.alert_system.active_alerts.items(), 
                               key=lambda item: (item[1]['type'] != 'Frame Exit', item[1]['type'] != 'Camera Range Exit', item[1]['start_time']), reverse=True)

        for alert_id, alert_data in sorted_alerts:
            obj_id_display = alert_data.get('object_id', 'N/A')
            class_name_display = alert_data.get('class_name', 'N/A')
            
            alert_text_prefix = ""
            if alert_data['type'] == 'threshold':
                distance = alert_data.get('distance', 0)
                alert_text_prefix = f"🚨 MOVEMENT ALERT: {class_name_display} #{obj_id_display} moved {distance:.1f}px beyond threshold!"
            elif alert_data['type'] == 'frame_exit':
                alert_text_prefix = f"🚨 FRAME EXIT ALERT: {class_name_display} #{obj_id_display} has left its reference frame!"
            elif alert_data['type'] == 'Camera Range Exit':
                alert_text_prefix = f"🔴 CAMERA RANGE ALERT!: {class_name_display} #{obj_id_display} left camera view!"
            elif alert_data['type'] == 'Frame Exit Removal': # Alert for when an object is removed after frame exit
                alert_text_prefix = f"🔴 REMOVED: {class_name_display} #{obj_id_display} exited frame for too long!"
            
            # Add countdown to alert text
            time_left = max(0, self.alert_system.alert_duration - (time.time() - alert_data['start_time']))
            alert_text = f"{alert_text_prefix} ({time_left:.1f}s left)"
            
            cv2.putText(frame, alert_text, (10, alert_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA) # Larger, bolder text
            alert_y -= 35 # More space for larger text
            active_alert_count += 1
        
        # General Status / Number of Active Tracks / Alerts (bottom-right)
        status_line = f"Tracks: {len(tracked_objects)} | Alerts: {active_alert_count} | FPS: {self.fps:.2f} | Det: {self.detection_time_ms:.1f}ms | Track: {self.tracking_time_ms:.1f}ms"
        text_size_status = cv2.getTextSize(status_line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(frame, status_line, (width - text_size_status[0] - 10, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA) # Yellow status text

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for object selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            tracked_objects = param # Param is self.tracker.objects
            
            # Find clicked object
            self.selected_object = None # Reset selection
            for obj_id, obj_data in tracked_objects.items():
                x1, y1, x2, y2 = obj_data['bbox']
                # Add a small buffer for easier clicking
                buffer = 10 
                if (x1 - buffer) <= x <= (x2 + buffer) and (y1 - buffer) <= y <= (y2 + buffer):
                    self.selected_object = obj_id
                    class_name = get_class_name(obj_data['class'])
                    print(f"✅ Selected {class_name} #{obj_id}")
                    break
    
    def process_keyboard(self, key, tracked_objects, frame_shape):
        """Process keyboard input for threshold adjustment and other controls."""
        height, width = frame_shape[:2]

        if key == ord('h') or key == ord('H'):
            self.show_help = not self.show_help
            print(f"Help {'shown' if self.show_help else 'hidden'}.")
        
        elif key == ord('r') or key == ord('R'):
            # Reset all thresholds and force ALL objects to re-settle
            for obj_data in tracked_objects.values():
                obj_data['threshold'] = DEFAULT_MOVEMENT_THRESHOLD
                
                # Force re-settle: this resets their baseline for movement
                obj_data['settle_start_time'] = time.time() 
                obj_data['has_settled'] = False 
                
                # Reset alert states for this object
                obj_data['alert_active'] = False
                obj_data['frame_alert_active'] = False
                obj_data['alert_debounce_counter'] = 0
                obj_data['frame_alert_debounce_counter'] = 0
                obj_data['mark_for_removal'] = False # Clear any pending removal
                obj_data['frame_exit_start_time'] = None # Clear frame exit timer
                self.alert_system.remove_object_alerts(obj_data['id'])


            print("🔄 All object thresholds and reference points reset. Objects will re-settle.")
        
        elif key == ord('s') or key == ord('S'):
            # Set current position as new origin/reference for selected object
            if self.selected_object and self.selected_object in tracked_objects:
                obj_data = tracked_objects[self.selected_object]
                
                # Force re-settle for selected object: this resets its baseline for movement
                obj_data['settle_start_time'] = time.time() 
                obj_data['has_settled'] = False 

                # Reset alerts for this object
                obj_data['alert_active'] = False
                obj_data['frame_alert_active'] = False
                obj_data['alert_debounce_counter'] = 0
                obj_data['frame_alert_debounce_counter'] = 0
                obj_data['mark_for_removal'] = False # Clear any pending removal
                obj_data['frame_exit_start_time'] = None # Clear frame exit timer
                self.alert_system.remove_object_alerts(self.selected_object)

                class_name = get_class_name(obj_data['class'])
                print(f"📍 New origin requested for {class_name} #{self.selected_object}. Object will re-settle.")

        elif key == ord('+') or key == ord('='):
            if self.selected_object and self.selected_object in tracked_objects:
                obj_data = tracked_objects[self.selected_object]
                obj_data['threshold'] += self.threshold_adjustment
                class_name = get_class_name(obj_data['class'])
                print(f"📈 Increased threshold for {class_name} #{self.selected_object} to {obj_data['threshold']}px")
        
        elif key == ord('-') or key == ord('_'):
            if self.selected_object and self.selected_object in tracked_objects:
                obj_data = tracked_objects[self.selected_object]
                obj_data['threshold'] = max(MIN_MOVEMENT_THRESHOLD, obj_data['threshold'] - self.threshold_adjustment)
                class_name = get_class_name(obj_data['class'])
                print(f"📉 Decreased threshold for {class_name} #{self.selected_object} to {obj_data['threshold']}px")
    
    def run(self, source=0):
        """Main execution loop."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f" Error: Could not open video source '{source}'")
            return
        
        # Set camera resolution (these might not take effect on all cameras/drivers)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        cv2.namedWindow('Real-Time Object Monitoring', cv2.WINDOW_NORMAL)
        # Set initial window size based on a typical frame or default large size
        initial_ret, initial_frame = cap.read()
        if initial_ret:
            h, w = initial_frame.shape[:2]
            # Ensure window size is reasonable and adds space for UI
            cv2.resizeWindow('Real-Time Object Monitoring', min(w + 400, 1920), min(h + 100, 1080)) 
        else:
            cv2.resizeWindow('Real-Time Object Monitoring', 1400, 900)
            print("Warning: Could not read initial frame to set window size dynamically. Using default.")

        print(" System running! Press 'H' for help, 'ESC' to exit")
        
        # Set mouse callback outside the loop to avoid re-setting every frame
        # We pass self.tracker.objects directly to the callback for it to modify the dict
        cv2.setMouseCallback('Real-Time Object Monitoring', self.mouse_callback, self.tracker.objects)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(" Error: Could not read frame. Exiting.")
                    break
                
                # --- Performance measurement (start) ---
                frame_start_time = time.time()

                # Run YOLO detection
                detection_start_time = time.time()
                # Suppress YOLO's internal print statements with verbose=False
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                detection_end_time = time.time()
                self.detection_time_ms = (detection_end_time - detection_start_time) * 1000
                
                # Extract detections
                detections = []
                if results and results[0].boxes is not None: # Ensure results exist and boxes are present
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    
                    for box, conf, cls in zip(boxes, confidences, classes):
                        detections.append([box[0], box[1], box[2], box[3], conf, cls])
                
                # Update tracker
                tracking_start_time = time.time()
                tracked_objects = self.tracker.update(detections, frame.shape)
                tracking_end_time = time.time()
                self.tracking_time_ms = (tracking_end_time - tracking_start_time) * 1000
                
                # Check for alerts
                self.alert_system.check_alerts(tracked_objects)
                
                # Draw UI
                self.draw_ui(frame, tracked_objects)
                
                # Display frame
                cv2.imshow('Real-Time Object Monitoring', frame)
                
                # --- Performance measurement (end) ---
                self.frame_count += 1
                current_total_time = time.time()
                elapsed_total_time = current_total_time - self.start_time
                if elapsed_total_time > 1.0: # Update FPS every second
                    self.fps = self.frame_count / elapsed_total_time
                    self.frame_count = 0
                    self.start_time = current_total_time

                # Process keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
                elif key != 255:  # Any other key (not an empty key press)
                    self.process_keyboard(key, tracked_objects, frame.shape)
        
        except KeyboardInterrupt:
            print("\n System stopped by user.")
        except Exception as e:
            print(f"\n An error occurred: {e}")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("Cleanup completed.")

def main():
    """Main function to run the monitoring system."""
    print("============================================")
    print(" Real-Time Object Monitoring and Alert System")
    print("============================================")
    
    # Initialize and run the system
    monitor = RealTimeMonitoringSystem(
        model_path='yolov8x.pt',  # Will download if not present, 'yolov8n.pt' for faster inference
        confidence_threshold=0.5 # Adjust this value (0.0 to 1.0) to control detection sensitivity
    )
    
    # Run with default camera (0) or change to video file path
    monitor.run(source=0)  # Use source='path/to/video.mp4' for video file or 0 for default camera

if __name__ == "__main__":
    main()