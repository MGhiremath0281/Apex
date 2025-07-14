import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import threading 

# --- Configuration ---
# Path to your trained YOLOv8 bird detection model
YOLO_MODEL_PATH = r'c:\Users\adity\Desktop\OBJDETECT\yolov8x.pt' 

# Camera source (0 for default webcam, or path to video file, or IP camera stream URL)
CAMERA_SOURCE = 0 

# Bird class ID (IMPORTANT: This MUST match the index of 'bird' in your model's class names)
BIRD_CLASS_NAME = 'bird' # We'll find the ID dynamically
BIRD_CLASS_ID = -1 # Initialize, will be set after model loads

# Tracking parameters (DeepSORT specific, adjust as needed)
MAX_TRACKING_AGE = 30 
MIN_HITS = 3         

# Avoidance parameters (tune carefully!)
CRITICAL_PROXIMITY_THRESHOLD_METERS = 50  
WARNING_PROXIMITY_THRESHOLD_METERS = 150 
COLLISION_TTC_THRESHOLD_SECONDS = 3     

# On-screen alert display settings
ALERT_DISPLAY_DURATION_SECONDS = 2 
ALERT_TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
ALERT_TEXT_SCALE = 1.5
ALERT_TEXT_THICKNESS = 3
ALERT_TEXT_COLOR = (255, 255, 255) # White
ALERT_BACKGROUND_COLOR = (0, 0, 255) # Red

# --- Main Detection and Tracking Logic ---

def run_bird_avoidance_system():
    # Load YOLOv8 model
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"YOLOv8 model loaded from {YOLO_MODEL_PATH}")

        # Dynamically find the bird class ID - FIX APPLIED HERE
        global BIRD_CLASS_ID
        found_bird_id = -1
        for class_id, class_name in model.names.items():
            # Make comparison robust: strip whitespace and convert to lowercase
            if class_name.strip().lower() == BIRD_CLASS_NAME.lower():
                found_bird_id = class_id
                break # Found it, no need to continue searching
        
        if found_bird_id != -1:
            BIRD_CLASS_ID = found_bird_id
            print(f"Found '{BIRD_CLASS_NAME}' class with ID: {BIRD_CLASS_ID}")
        else:
            print(f"Error: '{BIRD_CLASS_NAME}' not found in model's class names. "
                  "This might be due to a typo in BIRD_CLASS_NAME or a non-standard model.")
            print(f"Available classes: {model.names}")
            return # Exit if bird class not found

    except Exception as e:
        print(f"Error loading YOLO model: {e}. Please check path and file.")
        return

    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source {CAMERA_SOURCE}. "
              "Check if camera is connected, in use by another app, or if IP stream URL is correct.")
        return

    # --- Initialize Tracking (e.g., DeepSORT) ---
    tracks = {} 
    next_track_id = 0
    frame_count = 0
    fps_start_time = time.time()
    fps_frame_count = 0

    print("Starting real-time bird detection...")

    current_alert_message = "" 
    alert_display_start_time = 0 

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame or end of video stream.")
            break

        frame_count += 1
        fps_frame_count += 1

        # YOLOv8 Inference
        # Use the dynamically found BIRD_CLASS_ID for filtering detections
        results = model(frame, verbose=False, conf=0.5, classes=BIRD_CLASS_ID) 

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                # This check should now be redundant if 'classes=BIRD_CLASS_ID' works as intended,
                # but it adds an extra layer of safety.
                if cls == BIRD_CLASS_ID: 
                    detections.append({'bbox': [x1, y1, x2, y2], 'conf': conf})

        # --- Simple Centroid Tracking (Replace with DeepSORT for production) ---
        new_tracks = {}
        matched_detections = [False] * len(detections)

        for track_id, track_info in tracks.items():
            best_match_idx = -1
            min_dist = float('inf')
            
            track_centroid = track_info['centroid']

            for i, det in enumerate(detections):
                if matched_detections[i]:
                    continue
                det_centroid = ((det['bbox'][0] + det['bbox'][2]) // 2, (det['bbox'][1] + det['bbox'][3]) // 2)
                dist = np.linalg.norm(np.array(track_centroid) - np.array(det_centroid))

                if dist < 100 and dist < min_dist: 
                    min_dist = dist
                    best_match_idx = i
            
            if best_match_idx != -1:
                matched_detections[best_match_idx] = True
                matched_det = detections[best_match_idx]
                prev_centroid = track_info['centroid']
                current_centroid = ((matched_det['bbox'][0] + matched_det['bbox'][2]) // 2, (matched_det['bbox'][1] + matched_det['bbox'][3]) // 2)
                
                vx = current_centroid[0] - prev_centroid[0]
                vy = current_centroid[1] - prev_centroid[1]
                
                track_info['bbox'] = matched_det['bbox']
                track_info['centroid'] = current_centroid
                track_info['last_seen'] = frame_count
                track_info['velocity'] = (vx, vy)
                track_info['history'].append(current_centroid)
                if len(track_info['history']) > 10: 
                    track_info['history'].popleft()
                new_tracks[track_id] = track_info
        
        for i, det in enumerate(detections):
            if not matched_detections[i]:
                centroid = ((det['bbox'][0] + det['bbox'][2]) // 2, (det['bbox'][1] + det['bbox'][3]) // 2)
                new_tracks[next_track_id] = {
                    'bbox': det['bbox'],
                    'centroid': centroid,
                    'last_seen': frame_count,
                    'velocity': (0, 0), 
                    'history': deque([centroid])
                }
                next_track_id += 1
        
        tracks_to_remove = [tid for tid, info in tracks.items() if (frame_count - info['last_seen']) > MAX_TRACKING_AGE]
        for tid in tracks_to_remove:
            del tracks[tid]
        
        tracks = new_tracks 

        # --- Collision Prediction & Avoidance Logic ---
        collision_imminent = False
        warning_active = False

        for track_id, track_info in tracks.items():
            x1, y1, x2, y2 = track_info['bbox']
            centroid_x, centroid_y = track_info['centroid']
            vx, vy = track_info['velocity']

            bird_height_pixels = y2 - y1
            estimated_distance_meters = (1000 / (bird_height_pixels + 1)) 

            if estimated_distance_meters < CRITICAL_PROXIMITY_THRESHOLD_METERS:
                collision_imminent = True
            elif estimated_distance_meters < WARNING_PROXIMITY_THRESHOLD_METERS:
                warning_active = True

            if vx != 0 or vy != 0:
                relative_pos_x = centroid_x - (frame.shape[1] // 2)
                relative_pos_y = centroid_y - (frame.shape[0] // 2)
                
                if (vx * relative_pos_x + vy * relative_pos_y) < 0: 
                    ttc_x = -relative_pos_x / vx if vx != 0 else float('inf')
                    ttc_y = -relative_pos_y / vy if vy != 0 else float('inf')
                    ttc = min(abs(ttc_x), abs(ttc_y)) 
                    
                    if ttc > 0 and ttc < COLLISION_TTC_THRESHOLD_SECONDS * cap.get(cv2.CAP_PROP_FPS): 
                        collision_imminent = True

            # --- Visualization of Bird Detections ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Bird {track_id} ({int(estimated_distance_meters)}m)', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
            
            cv2.arrowedLine(frame, (centroid_x, centroid_y), 
                            (int(centroid_x + vx * 5), int(centroid_y + vy * 5)), 
                            (255, 0, 0), 2)
            
            for i in range(1, len(track_info['history'])):
                cv2.line(frame, track_info['history'][i-1], track_info['history'][i], (0, 255, 255), 1)

        # --- On-Screen Alert Logic ---
        new_alert_set = False
        if collision_imminent:
            if current_alert_message != "CRITICAL: COLLISION IMMINENT!":
                current_alert_message = "CRITICAL: COLLISION IMMINENT!"
                alert_display_start_time = time.time()
                new_alert_set = True
        elif warning_active and not collision_imminent: 
            if current_alert_message != "WARNING: Bird detected close!":
                current_alert_message = "WARNING: Bird detected close!"
                alert_display_start_time = time.time()
                new_alert_set = True
        
        if current_alert_message and (time.time() - alert_display_start_time >= ALERT_DISPLAY_DURATION_SECONDS) and not (collision_imminent or warning_active):
            current_alert_message = "" 

        # Display alert message if active
        if current_alert_message:
            text_size = cv2.getTextSize(current_alert_message, ALERT_TEXT_FONT, ALERT_TEXT_SCALE, ALERT_TEXT_THICKNESS)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2 
            text_y = 70 

            cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10), 
                          (text_x + text_size[0] + 10, text_y + 10), 
                          ALERT_BACKGROUND_COLOR, -1) 

            cv2.putText(frame, current_alert_message, (text_x, text_y), 
                        ALERT_TEXT_FONT, ALERT_TEXT_SCALE, ALERT_TEXT_COLOR, ALERT_TEXT_THICKNESS) 

        # Calculate and display FPS
        if time.time() - fps_start_time >= 1.0:
            fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display frame
        cv2.imshow('Bird Detection & Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("--- Bird Avoidance System Initializing ---")
    print("Ensure your camera source and model path are correctly configured.")
    print("All alerts will be displayed on screen.")
    print("Press 'q' to quit the display window.")
    run_bird_avoidance_system()
    print("--- System Shut Down ---")