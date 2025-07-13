import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
from playsound import playsound # <--- Changed this line
import threading
from pymavlink import mavutil 
import platform 

# --- Configuration ---
# Path to your trained YOLOv8 bird detection model
YOLO_MODEL_PATH = 'path/to/your/best_bird_model.pt' 

# Camera source (0 for default webcam, or path to video file, or IP camera stream URL)
CAMERA_SOURCE = 0 
# For a real system, this would be a high-res, low-latency camera stream
# Example for IP camera: 'rtsp://your_camera_ip/stream'

# Bird class ID (make sure this matches your model's training)
BIRD_CLASS_ID = 0 # Example: if 'bird' is the first class in your model's classes.txt

# Tracking parameters (DeepSORT specific, adjust as needed)
MAX_TRACKING_AGE = 30 # How many frames to keep a track without detection
MIN_HITS = 3         # How many detections before a track is confirmed

# Avoidance parameters (tune carefully!)
CRITICAL_PROXIMITY_THRESHOLD_METERS = 50  # Distance at which a bird is considered critical
WARNING_PROXIMITY_THRESHOLD_METERS = 150 # Distance at which a bird triggers a warning
COLLISION_TTC_THRESHOLD_SECONDS = 3     # Time-To-Collision (TTC) to trigger evasive action

# Audio alert paths (replace with actual alert sounds)
WARNING_AUDIO_PATH = 'path/to/warning_alert.wav'
CRITICAL_AUDIO_PATH = 'path/to/critical_alert.wav'

# MAVLink connection details (for drone integration)
MAVLINK_CONNECTION_STRING = 'udp:127.0.0.1:14550' # Example for SITL or local UDP
# For a physical drone, this would be 'serial:/dev/ttyACM0:57600' or similar
MAVLINK_TARGET_SYSTEM = 1
MAVLINK_TARGET_COMPONENT = 1

# --- Global Variables for MAVLink (if used) ---
master = None
mavlink_connected = False

# --- Helper Functions ---

def play_audio_alert(file_path):
    """Plays a WAV audio file for an alert using playsound."""
    try:
        # playsound by default blocks, but since we call it in a thread, it's fine.
        # block=False might be needed for some older versions or specific OS,
        # but often it works fine within a thread without it.
        playsound(file_path) # <--- Changed this line
    except Exception as e:
        print(f"Error playing audio {file_path}: {e}")

def initialize_mavlink():
    """Initializes MAVLink connection."""
    global master, mavlink_connected
    try:
        print(f"Attempting to connect to MAVLink: {MAVLINK_CONNECTION_STRING}")
        master = mavutil.mavlink_connection(MAVLINK_CONNECTION_STRING)
        master.wait_heartbeat()
        print(f"MAVLink connected to system {master.target_system} component {master.target_component}")
        master.target_system = MAVLINK_TARGET_SYSTEM
        master.target_component = MAVLINK_TARGET_COMPONENT
        mavlink_connected = True
    except Exception as e:
        print(f"MAVLink connection failed: {e}")
        mavlink_connected = False

def send_mavlink_command(command_type, **kwargs):
    """Sends a MAVLink command (simplified example)."""
    if not mavlink_connected:
        print("MAVLink not connected, cannot send command.")
        return

    try:
        if command_type == "VELOCITY_CHANGE":
            vx = kwargs.get('vx', 0.0)
            vy = kwargs.get('vy', 0.0)
            vz = kwargs.get('vz', 0.0)
            print(f"MAVLink: Sending velocity command: vx={vx}, vy={vy}, vz={vz}")
            pass 
        elif command_type == "ALERT_TEXT":
            text = kwargs.get('text', 'Bird Alert!')
            master.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_INFO, text.encode('utf8'))
            print(f"MAVLink: Sent text alert: '{text}'")

    except Exception as e:
        print(f"Error sending MAVLink command: {e}")

# --- Main Detection and Tracking Logic ---

def run_bird_avoidance_system():
    global master, mavlink_connected

    # Load YOLOv8 model
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"YOLOv8 model loaded from {YOLO_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading YOLO model: {e}. Please check path and file.")
        return

    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source {CAMERA_SOURCE}.")
        return

    # --- Initialize Tracking (e.g., DeepSORT) ---
    tracks = {} 
    next_track_id = 0
    frame_count = 0
    fps_start_time = time.time()
    fps_frame_count = 0

    if not platform.system() == "Windows": 
         mavlink_thread = threading.Thread(target=initialize_mavlink)
         mavlink_thread.daemon = True 
         mavlink_thread.start()
    else:
        print("MAVLink thread not started on Windows (consider your setup).")


    print("Starting real-time bird detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame or end of video stream.")
            break

        frame_count += 1
        fps_frame_count += 1

        # YOLOv8 Inference
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
                        print(f"Bird {track_id}: TTC estimated at {ttc / cap.get(cv2.CAP_PROP_FPS):.2f} seconds!")

            # --- Visualization ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Bird {track_id} ({int(estimated_distance_meters)}m)', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
            
            cv2.arrowedLine(frame, (centroid_x, centroid_y), 
                            (int(centroid_x + vx * 5), int(centroid_y + vy * 5)), 
                            (255, 0, 0), 2)
            
            for i in range(1, len(track_info['history'])):
                cv2.line(frame, track_info['history'][i-1], track_info['history'][i], (0, 255, 255), 1)

        # --- Trigger Actions ---
        if collision_imminent:
            threading.Thread(target=play_audio_alert, args=(CRITICAL_AUDIO_PATH,)).start()
            print("CRITICAL ALERT: COLLISION IMMINENT!")
            if mavlink_connected:
                send_mavlink_command("ALERT_TEXT", text="CRITICAL: BIRD COLLISION")
        elif warning_active:
            threading.Thread(target=play_audio_alert, args=(WARNING_AUDIO_PATH,)).start()
            print("WARNING: Bird detected close!")
            if mavlink_connected:
                send_mavlink_command("ALERT_TEXT", text="WARNING: Bird nearby")

        # Calculate and display FPS
        if time.time() - fps_start_time >= 1.0:
            fps = fps_frame_count / (time.time() - fps_start_time)
            print(f"FPS: {fps:.2f}")
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
    if master:
        master.close() 

if __name__ == '__main__':
    print("--- Bird Avoidance System Initializing ---")
    print("Ensure your camera source and model path are correctly configured.")
    print("For MAVLink, ensure your connection string matches your flight controller/SITL.")
    print("Press 'q' to quit the display window.")
    run_bird_avoidance_system()
    print("--- System Shut Down ---")