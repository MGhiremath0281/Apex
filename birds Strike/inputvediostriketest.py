import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import threading 

# --- Configuration ---
# Path to your trained YOLOv8 bird detection model
YOLO_MODEL_PATH = r'c:\Users\adity\Desktop\OBJDETECT\yolov8x.pt' 

# Video source: Provide the full path to your video file
# IMPORTANT: Replace 'path/to/your/video.mp4' with the actual path to your video.
# Example: VIDEO_PATH = r'c:\Users\adity\Desktop\OBJDETECT\my_bird_video.mp4'
VIDEO_PATH = r'c:\Users\adity\Desktop\OBJDETECT\data\birds2.mp4' # <--- CHANGE THIS TO YOUR VIDEO FILE PATH

CAMERA_SOURCE = VIDEO_PATH # Set camera source to the video file

# Bird class name (make sure this matches your model's training)
BIRD_CLASS_NAME = 'bird' 
BIRD_CLASS_ID = -1 # Will be set after model loads

# Tracking parameters (DeepSORT specific, adjust as needed)
MAX_TRACKING_AGE = 30 
MIN_HITS = 3         

# Avoidance parameters (tune carefully!)
CRITICAL_PROXIMITY_THRESHOLD_METERS = 50  
WARNING_PROXIMITY_THRESHOLD_METERS = 150 
COLLISION_TTC_THRESHOLD_SECONDS = 3     

# Region of Interest (ROI) for testing
# Define your ROI coordinates (x1, y1, x2, y2)
# Adjust these based on the resolution of your video and the area you want to monitor.
# Example: For a 1280x720 video, this might cover a central area.
ROI_X1 = 300
ROI_Y1 = 200
ROI_X2 = 900
ROI_Y2 = 600
ROI_COLOR = (0, 255, 255) # Yellow
ROI_THICKNESS = 2

# On-screen alert display settings
ALERT_DISPLAY_DURATION_SECONDS = 2 
ALERT_TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
ALERT_TEXT_SCALE = 1.5
ALERT_TEXT_THICKNESS = 3
ALERT_TEXT_COLOR = (255, 255, 255) # White
ALERT_BACKGROUND_COLOR_CRITICAL = (0, 0, 255) # Red for collision
ALERT_BACKGROUND_COLOR_WARNING = (0, 165, 255) # Orange for warning
ALERT_BACKGROUND_COLOR_ROI = (255, 0, 0) # Blue for ROI violation

# --- Main Detection and Tracking Logic ---

def run_bird_avoidance_system():
    # Load YOLOv8 model
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"YOLOv8 model loaded from {YOLO_MODEL_PATH}")

        # Dynamically find the bird class ID
        global BIRD_CLASS_ID
        found_bird_id = -1
        for class_id, class_name in model.names.items():
            if class_name.strip().lower() == BIRD_CLASS_NAME.lower():
                found_bird_id = class_id
                break
        
        if found_bird_id != -1:
            BIRD_CLASS_ID = found_bird_id
            print(f"Found '{BIRD_CLASS_NAME}' class with ID: {BIRD_CLASS_ID}")
        else:
            print(f"Error: '{BIRD_CLASS_NAME}' not found in model's class names. "
                  "Please check BIRD_CLASS_NAME or your model's training classes.")
            print(f"Available classes: {model.names}")
            return 

    except Exception as e:
        print(f"Error loading YOLO model: {e}. Please check path and file.")
        return

    # Initialize video capture from file
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {CAMERA_SOURCE}. "
              "Check if the path is correct, file exists, and video codecs are installed.")
        return

    # Get video properties for potential use (e.g., frame dimensions)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video loaded: {CAMERA_SOURCE} ({frame_width}x{frame_height} @ {fps_video:.2f} FPS)")


    # --- Initialize Tracking (e.g., Simple Centroid Tracking) ---
    tracks = {} 
    next_track_id = 0
    frame_count = 0
    fps_start_time = time.time()
    fps_frame_count = 0

    print("Starting video analysis for bird detection...")

    current_alert_message = "" 
    alert_display_start_time = 0 
    alert_background_color = (0,0,0) # Default transparent/black

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or failed to grab frame.")
            break # Exit loop when video ends

        frame_count += 1
        fps_frame_count += 1

        # Draw ROI on the frame
        cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), ROI_COLOR, ROI_THICKNESS)
        cv2.putText(frame, 'ROI', (ROI_X1, ROI_Y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ROI_COLOR, 2)

        # YOLOv8 Inference - only detect the bird class
        results = model(frame, verbose=False, conf=0.5, classes=BIRD_CLASS_ID) 

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if cls == BIRD_CLASS_ID: 
                    detections.append({'bbox': [x1, y1, x2, y2], 'conf': conf})

        # --- Simple Centroid Tracking ---
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
        bird_in_roi = False 

        for track_id, track_info in tracks.items():
            x1, y1, x2, y2 = track_info['bbox']
            centroid_x, centroid_y = track_info['centroid']
            vx, vy = track_info['velocity']

            bird_height_pixels = y2 - y1
            estimated_distance_meters = (1000 / (bird_height_pixels + 1)) 

            # Check for Proximity Alerts
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

            # --- Check if bird is within ROI ---
            # Check for overlap between bird's bounding box and ROI
            if (x1 < ROI_X2 and x2 > ROI_X1 and
                y1 < ROI_Y2 and y2 > ROI_Y1):
                bird_in_roi = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), ALERT_BACKGROUND_COLOR_ROI, 2) # Highlight bird in ROI
                cv2.putText(frame, 'IN ROI', (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ALERT_BACKGROUND_COLOR_ROI, 2)


            # --- Visualization of Bird Detections ---
            # Only draw green bounding box if not in ROI (ROI highlights override)
            if not bird_in_roi:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.putText(frame, f'Bird {track_id} ({int(estimated_distance_meters)}m)', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
            
            cv2.arrowedLine(frame, (centroid_x, centroid_y), 
                            (int(centroid_x + vx * 5), int(centroid_y + vy * 5)), 
                            (255, 0, 0), 2)
            
            for i in range(1, len(track_info['history'])):
                cv2.line(frame, track_info['history'][i-1], track_info['history'][i], (0, 255, 255), 1)

        # --- On-Screen Alert Logic (Hierarchy: ROI > Critical > Warning) ---
        new_alert_set = False
        temp_alert_message = ""
        temp_alert_color = (0,0,0)

        if bird_in_roi:
            temp_alert_message = "BIRD IN RESTRICTED AREA!"
            temp_alert_color = ALERT_BACKGROUND_COLOR_ROI # Blue background for ROI
        elif collision_imminent:
            temp_alert_message = "CRITICAL: COLLISION IMMINENT!"
            temp_alert_color = ALERT_BACKGROUND_COLOR_CRITICAL # Red background
        elif warning_active:
            temp_alert_message = "WARNING: Bird detected close!"
            temp_alert_color = ALERT_BACKGROUND_COLOR_WARNING # Orange background
        
        # Update current alert if a higher priority alert is active or existing alert is timed out
        if temp_alert_message != current_alert_message: 
            current_alert_message = temp_alert_message
            alert_background_color = temp_alert_color
            alert_display_start_time = time.time()
        elif current_alert_message and (time.time() - alert_display_start_time >= ALERT_DISPLAY_DURATION_SECONDS):
            if not (bird_in_roi or collision_imminent or warning_active):
                current_alert_message = ""
                alert_background_color = (0,0,0) 

        # Display alert message if active
        if current_alert_message:
            text_size = cv2.getTextSize(current_alert_message, ALERT_TEXT_FONT, ALERT_TEXT_SCALE, ALERT_TEXT_THICKNESS)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2 
            text_y = 70 

            cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10), 
                          (text_x + text_size[0] + 10, text_y + 10), 
                          alert_background_color, -1) 

            cv2.putText(frame, current_alert_message, (text_x, text_y), 
                        ALERT_TEXT_FONT, ALERT_TEXT_SCALE, ALERT_TEXT_COLOR, ALERT_TEXT_THICKNESS) 

        # Calculate and display FPS
        if time.time() - fps_start_time >= 1.0:
            fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display frame
        cv2.imshow('Bird Detection & Tracking (Video Test Mode)', frame)

        # Control video playback speed (optional, adjust as needed)
        # For real-time playback, use a small delay based on video FPS
        wait_time = int(1000 / fps_video) if fps_video > 0 else 1
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("--- Bird Avoidance System (Video Test Mode) Initializing ---")
    print("Video Source: ", VIDEO_PATH)
    print("Ensure your model path is correctly configured.")
    print("ROI: [X1, Y1, X2, Y2] = [", ROI_X1, ", ", ROI_Y1, ", ", ROI_X2, ", ", ROI_Y2, "]")
    print("Alerts will be displayed on screen.")
    print("Press 'q' to quit the display window.")
    run_bird_avoidance_system()
    print("--- System Shut Down ---")