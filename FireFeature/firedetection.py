import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import os  # For creating log directory if needed

# --- Configuration ---
# Paths to your YOLO models
BASE_YOLO_MODEL_PATH = r'c:\Users\adity\Desktop\OBJDETECT\yolov8x.pt'  # Standard YOLOv8 model
FIRE_YOLO_MODEL_PATH = r'c:\Users\adity\Desktop\OBJDETECT\yolofirenew.pt'  # <--- IMPORTANT: SET PATH TO YOUR YOLOfire MODEL

# Video source (e.g., webcam 0, or path to video file)
VIDEO_SOURCE = r'c:\Users\adity\Desktop\OBJDETECT\data\firedata2.mp4'  # <--- CHANGE THIS TO YOUR VIDEO FILE PATH

# Class names to detect from BASE_YOLO_MODEL_PATH
HUMAN_CLASS_NAME = 'person'
# List of vehicle class names from COCO dataset
VEHICLE_CLASS_NAMES = ['car', 'truck', 'bus', 'motorcycle']

# Class name for fire from FIRE_YOLO_MODEL_PATH
# IMPORTANT: Adjust 'Fire' if your YOLOfire model uses a different name (e.g., 'flame', 'smoke')
FIRE_CLASS_NAME = 'Fire'  # <-- FIXED CASE TO MATCH YOUR MODEL

# Confidence threshold for detections
CONFIDENCE_THRESHOLD_BASE = 0.5  # For human/vehicles
CONFIDENCE_THRESHOLD_FIRE = 0.3  # For fire (adjust as needed)

# Class IDs - will be found dynamically after models load
HUMAN_CLASS_ID = -1
VEHICLE_CLASS_IDS = []  # List to store IDs of specified vehicles
FIRE_CLASS_ID = -1

# On-screen alert display settings
ALERT_TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
ALERT_TEXT_SCALE = 1.5
ALERT_TEXT_THICKNESS = 3
ALERT_TEXT_COLOR = (255, 255, 255)  # White
ALERT_BACKGROUND_COLOR_FIRE = (0, 0, 255)  # Bright Red for Fire!
ALERT_BACKGROUND_COLOR_HUMAN = (255, 0, 0)  # Blue for Human detected
ALERT_BACKGROUND_COLOR_VEHICLE = (0, 255, 255)  # Yellow for Vehicle detected

# Logging configuration
LOGS_DIRECTORY = 'detection_logs'
LOG_FILE_NAME_PREFIX = 'detection_log_'
LOG_FILE = None  # Will be opened dynamically
CURRENT_FRAME_NUMBER = 0  # To track frame number for logging


# --- Logging Function ---
def setup_logging():
    global LOG_FILE
    # Create logs directory if it doesn't exist
    if not os.path.exists(LOGS_DIRECTORY):
        os.makedirs(LOGS_DIRECTORY)
        print(f"Created logging directory: {LOGS_DIRECTORY}")

    # Generate a unique log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(LOGS_DIRECTORY, f"{LOG_FILE_NAME_PREFIX}{timestamp}.txt")

    try:
        LOG_FILE = open(log_file_path, 'a')  # Open in append mode
        LOG_FILE.write(f"--- Detection Log Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        LOG_FILE.write(f"Video Source: {VIDEO_SOURCE}\n")
        LOG_FILE.write(f"Base Model: {os.path.basename(BASE_YOLO_MODEL_PATH)}\n")
        LOG_FILE.write(f"Fire Model: {os.path.basename(FIRE_YOLO_MODEL_PATH)}\n")
        LOG_FILE.write("-" * 50 + "\n")
        print(f"Logging to: {log_file_path}")
    except Exception as e:
        print(f"Error setting up log file: {e}")
        LOG_FILE = None  # Ensure LOG_FILE is None if opening fails


def log_event(message):
    global LOG_FILE
    if LOG_FILE:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Milliseconds
        LOG_FILE.write(f"[{current_time} | Frame {CURRENT_FRAME_NUMBER}] {message}\n")
        LOG_FILE.flush()  # Ensure immediate write to disk


# --- Main Detection Logic ---
def run_combined_detection():
    global HUMAN_CLASS_ID, VEHICLE_CLASS_IDS, FIRE_CLASS_ID, CURRENT_FRAME_NUMBER

    # Setup logging at the very beginning
    setup_logging()
    if LOG_FILE is None:
        print("Logging is disabled due to file setup error.")

    # Load YOLO models
    try:
        base_model = YOLO(BASE_YOLO_MODEL_PATH)
        fire_model = YOLO(FIRE_YOLO_MODEL_PATH)
        print(f"Base YOLO model loaded from {BASE_YOLO_MODEL_PATH}")
        print(f"YOLOfire model loaded from {FIRE_YOLO_MODEL_PATH}")
        log_event(f"Base YOLO model loaded: {os.path.basename(BASE_YOLO_MODEL_PATH)}")
        log_event(f"YOLOfire model loaded: {os.path.basename(FIRE_YOLO_MODEL_PATH)}")

        # Dynamically find class IDs for base model
        if HUMAN_CLASS_NAME.lower() in (name.lower() for name in base_model.names.values()):
            HUMAN_CLASS_ID = [k for k, v in base_model.names.items() if v.lower() == HUMAN_CLASS_NAME.lower()][0]
            print(f"Found '{HUMAN_CLASS_NAME}' class ID: {HUMAN_CLASS_ID} in base model.")
        else:
            print(f"WARNING: '{HUMAN_CLASS_NAME}' not found in base model's classes. Human detection will not work.")
            log_event(f"WARNING: '{HUMAN_CLASS_NAME}' not found in base model classes.")

        for vehicle_name in VEHICLE_CLASS_NAMES:
            if vehicle_name.lower() in (name.lower() for name in base_model.names.values()):
                VEHICLE_CLASS_IDS.append([k for k, v in base_model.names.items() if v.lower() == vehicle_name.lower()][0])
                print(f"Found '{vehicle_name}' class ID: {VEHICLE_CLASS_IDS[-1]} in base model.")
            else:
                print(f"WARNING: '{vehicle_name}' not found in base model's classes. This vehicle type will not be detected.")
                log_event(f"WARNING: '{vehicle_name}' not found in base model classes.")

        # Dynamically find class ID for fire model (case-insensitive)
        if FIRE_CLASS_NAME.lower() in (name.lower() for name in fire_model.names.values()):
            FIRE_CLASS_ID = [k for k, v in fire_model.names.items() if v.lower() == FIRE_CLASS_NAME.lower()][0]
            print(f"Found '{FIRE_CLASS_NAME}' class ID: {FIRE_CLASS_ID} in fire model.")
            log_event(f"Found '{FIRE_CLASS_NAME}' class ID: {FIRE_CLASS_ID} in fire model.")
        else:
            print(f"ERROR: '{FIRE_CLASS_NAME}' not found in YOLOfire model's classes. Fire detection will NOT work.")
            print(f"Available classes in YOLOfire model: {fire_model.names}")
            log_event(f"ERROR: '{FIRE_CLASS_NAME}' not found in YOLOfire model classes. Available: {fire_model.names}")
            if LOG_FILE:
                LOG_FILE.close()
            return  # Exit if primary fire detection fails

    except Exception as e:
        print(f"Error loading one or both YOLO models: {e}. Please check paths and file integrity.")
        log_event(f"FATAL ERROR: Failed to load models: {e}")
        if LOG_FILE:
            LOG_FILE.close()
        return

    # Initialize video capture
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}.")
        log_event(f"FATAL ERROR: Could not open video source {VIDEO_SOURCE}.")
        if LOG_FILE:
            LOG_FILE.close()
        return

    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0.0

    print("Starting combined real-time detection...")

    current_alert_message = ""
    alert_background_color = (0, 0, 0)
    alert_display_start_time = 0
    ALERT_DISPLAY_DURATION_SECONDS = 2  # Duration for alerts to stay on screen

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or failed to grab frame.")
            log_event("End of video stream or failed to grab frame.")
            break

        CURRENT_FRAME_NUMBER += 1
        fps_frame_count += 1

        all_detections = []  # To store combined detections

        # --- Inference with Base Model (Human, Vehicles) ---
        base_classes_to_detect = []
        if HUMAN_CLASS_ID != -1:
            base_classes_to_detect.append(HUMAN_CLASS_ID)
        base_classes_to_detect.extend(VEHICLE_CLASS_IDS)

        if base_classes_to_detect:
            base_results = base_model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD_BASE, classes=base_classes_to_detect)
            for r in base_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    all_detections.append({'bbox': [x1, y1, x2, y2], 'conf': conf, 'cls_id': cls, 'model': 'base'})

        # --- Inference with Fire Model (Fire) ---
        if FIRE_CLASS_ID != -1:
            fire_results = fire_model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD_FIRE, classes=FIRE_CLASS_ID)
            for r in fire_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    all_detections.append({'bbox': [x1, y1, x2, y2], 'conf': conf, 'cls_id': cls, 'model': 'fire'})

        # --- Process and Visualize Combined Detections ---
        fire_detected = False
        human_detected = False
        vehicle_detected = False

        for det in all_detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            cls_id = det['cls_id']
            model_source = det['model']

            label = ""
            color = (0, 0, 0)  # Default to black
            detection_type = "UNKNOWN"

            if model_source == 'fire' and cls_id == FIRE_CLASS_ID:
                label = f'FIRE {conf:.2f}'
                color = (0, 0, 255)  # Red for fire
                fire_detected = True
                detection_type = "FIRE"
            elif model_source == 'base':
                if cls_id == HUMAN_CLASS_ID:
                    label = f'{HUMAN_CLASS_NAME.capitalize()} {conf:.2f}'
                    color = (255, 0, 0)  # Blue for human
                    human_detected = True
                    detection_type = "HUMAN"
                elif cls_id in VEHICLE_CLASS_IDS:
                    vehicle_name = base_model.names[cls_id].capitalize()
                    label = f'{vehicle_name} {conf:.2f}'
                    color = (0, 255, 255)  # Yellow for vehicles
                    vehicle_detected = True
                    detection_type = "VEHICLE"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Log individual detection
            log_event(f"Detected {detection_type}: {label} (Conf: {conf:.2f}, BBox: [{x1}, {y1}, {x2}, {y2}])")

        # --- On-Screen Alert Logic (Hierarchy: FIRE > Human > Vehicle) ---
        temp_alert_message = ""
        temp_alert_color = (0, 0, 0)

        if fire_detected:
            temp_alert_message = "ALERT: FIRE DETECTED!"
            temp_alert_color = ALERT_BACKGROUND_COLOR_FIRE
        elif human_detected:
            temp_alert_message = "ALERT: HUMAN DETECTED!"
            temp_alert_color = ALERT_BACKGROUND_COLOR_HUMAN
        elif vehicle_detected:
            temp_alert_message = "ALERT: VEHICLE DETECTED!"
            temp_alert_color = ALERT_BACKGROUND_COLOR_VEHICLE

        # Check if alert state has changed for logging
        if temp_alert_message != current_alert_message:
            if current_alert_message:  # Log when an existing alert clears implicitly
                log_event(f"Alert Cleared: {current_alert_message}")
            if temp_alert_message:  # Log when a new alert is triggered
                log_event(f"Alert Triggered: {temp_alert_message}")

            current_alert_message = temp_alert_message
            alert_background_color = temp_alert_color
            alert_display_start_time = time.time()
        elif current_alert_message and (time.time() - alert_display_start_time >= ALERT_DISPLAY_DURATION_SECONDS):
            # Clear current alert if timed out and no active conditions
            if not (fire_detected or human_detected or vehicle_detected):
                log_event(f"Alert Cleared: {current_alert_message} (Timed out)")
                current_alert_message = ""
                alert_background_color = (0, 0, 0)

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

        # --- Calculate and display FPS ---
        if fps_frame_count >= 10:
            fps_end_time = time.time()
            time_diff = fps_end_time - fps_start_time
            fps = fps_frame_count / time_diff if time_diff > 0 else 0.0
            fps_start_time = fps_end_time
            fps_frame_count = 0

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Combined YOLOv8 Detection (Human, Vehicles, Fire)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to quit
            print("ESC pressed. Exiting...")
            log_event("ESC pressed by user. Exiting detection.")
            break

    cap.release()
    cv2.destroyAllWindows()
    if LOG_FILE:
        LOG_FILE.write(f"--- Detection Log Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        LOG_FILE.close()
        print("Log file closed.")


if __name__ == "__main__":
    run_combined_detection()
