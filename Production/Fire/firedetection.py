import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import os
import threading
from playsound import playsound

# --- Configurations ---
BASE_YOLO_MODEL_PATH = r'c:\Users\adity\Desktop\OBJDETECT\yolov8x.pt'
FIRE_YOLO_MODEL_PATH = r'c:\Users\adity\Desktop\OBJDETECT\yolofirenew.pt'
VIDEO_SOURCE = r'c:\Users\adity\Desktop\OBJDETECT\data\firedata2.mp4'

HUMAN_CLASS_NAME = 'person'
VEHICLE_CLASS_NAMES = ['car', 'truck', 'bus', 'motorcycle']
FIRE_CLASS_NAME = 'Fire'

CONFIDENCE_THRESHOLD_BASE = 0.5
CONFIDENCE_THRESHOLD_FIRE = 0.3

ALERT_TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
ALERT_TEXT_SCALE = 1.5
ALERT_TEXT_THICKNESS = 3
ALERT_TEXT_COLOR = (255, 255, 255)

ALERT_BACKGROUND_COLOR_FIRE = (0, 0, 255)
ALERT_BACKGROUND_COLOR_HUMAN = (255, 0, 0)
ALERT_BACKGROUND_COLOR_VEHICLE = (0, 255, 255)
ALERT_BACKGROUND_COLOR_BOTH = (255, 0, 255)

FIRE_SOUND_PATH = 'fire_alert.mp3'
HUMAN_SOUND_PATH = 'human_alert.mp3'
VEHICLE_SOUND_PATH = 'vehicle_alert.mp3'

IR_MODE_ENABLED = False
CURRENT_ROTATION_ANGLE = 0
CURRENT_FRAME_NUMBER = 0

# --- Cooldown Config ---
last_alert_type = ""
last_alert_time = 0
ALERT_COOLDOWN_SECONDS = 3

def log_event(message):
    global CURRENT_FRAME_NUMBER
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{now} | Frame {CURRENT_FRAME_NUMBER}] {message}")

def play_sound_async(path):
    if os.path.exists(path):
        threading.Thread(target=playsound, args=(path,), kwargs={'block': False}).start()
    else:
        log_event(f"Missing sound: {path}")

def trigger_sound_alert(alert_type):
    path = None
    if alert_type == "FIRE":
        path = FIRE_SOUND_PATH
    elif alert_type == "HUMAN":
        path = HUMAN_SOUND_PATH
    elif alert_type == "VEHICLE":
        path = VEHICLE_SOUND_PATH
    elif alert_type == "HUMAN_VEHICLE":
        path = HUMAN_SOUND_PATH  # You can set a custom combined alert if needed
    if path:
        play_sound_async(path)
        log_event(f"Sound alert triggered: {alert_type}")

def run_combined_detection():
    global CURRENT_FRAME_NUMBER, CURRENT_ROTATION_ANGLE, IR_MODE_ENABLED
    global last_alert_type, last_alert_time

    try:
        base_model = YOLO(BASE_YOLO_MODEL_PATH)
        fire_model = YOLO(FIRE_YOLO_MODEL_PATH)
        print("✅ Models loaded.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return


    HUMAN_CLASS_ID = [k for k, v in base_model.names.items() if v.lower() == HUMAN_CLASS_NAME.lower()][0]
    VEHICLE_CLASS_IDS = [k for k, v in base_model.names.items() if v.lower() in VEHICLE_CLASS_NAMES]
    FIRE_CLASS_ID = [k for k, v in fire_model.names.items() if v.lower() == FIRE_CLASS_NAME.lower()][0]

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"❌ Could not open video: {VIDEO_SOURCE}")
        return

    fps_frame_count, fps_start_time, fps = 0, time.time(), 0.0
    current_alert_message, alert_background_color = "", (0, 0, 0)
    alert_display_start_time = 0
    ALERT_DISPLAY_DURATION_SECONDS = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ End of video or read error.")
            break

        CURRENT_FRAME_NUMBER += 1
        fps_frame_count += 1

        # Apply rotation
        if CURRENT_ROTATION_ANGLE == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif CURRENT_ROTATION_ANGLE == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif CURRENT_ROTATION_ANGLE == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # IR Mode
        if IR_MODE_ENABLED:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)

        all_detections = []

        # Base Model
        base_classes = [HUMAN_CLASS_ID] + VEHICLE_CLASS_IDS
        base_results = base_model(frame, conf=CONFIDENCE_THRESHOLD_BASE, classes=base_classes, verbose=False)
        for result in base_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                all_detections.append({'bbox': [x1, y1, x2, y2], 'conf': conf, 'cls': cls, 'model': 'base'})

        # Fire Model
        fire_results = fire_model(frame, conf=CONFIDENCE_THRESHOLD_FIRE, classes=FIRE_CLASS_ID, verbose=False)
        for result in fire_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                all_detections.append({'bbox': [x1, y1, x2, y2], 'conf': conf, 'cls': cls, 'model': 'fire'})

        fire_detected = False
        human_detected = False
        vehicle_detected = False

        for det in all_detections:
            x1, y1, x2, y2 = det['bbox']
            conf, cls_id, model = det['conf'], det['cls'], det['model']

            if model == 'fire' and cls_id == FIRE_CLASS_ID:
                label = f'FIRE {conf:.2f}'
                color = (0, 0, 255)
                fire_detected = True
                detection_type = "FIRE"
            elif model == 'base':
                if cls_id == HUMAN_CLASS_ID:
                    label = f'Person {conf:.2f}'
                    color = (255, 0, 0)
                    human_detected = True
                    detection_type = "HUMAN"
                elif cls_id in VEHICLE_CLASS_IDS:
                    label = f'{base_model.names[cls_id].capitalize()} {conf:.2f}'
                    color = (0, 255, 255)
                    vehicle_detected = True
                    detection_type = "VEHICLE"
                else:
                    continue
            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            log_event(f"{detection_type}: {label} at [{x1}, {y1}, {x2}, {y2}]")

        # Decide Alert Type
        current_time_sec = time.time()
        temp_alert_message = ""
        temp_alert_color = (0, 0, 0)
        alert_type = None

        if fire_detected:
            temp_alert_message = "ALERT: FIRE DETECTED!"
            temp_alert_color = ALERT_BACKGROUND_COLOR_FIRE
            alert_type = "FIRE"
        elif human_detected and vehicle_detected:
            temp_alert_message = "ALERT: HUMAN & VEHICLE DETECTED!"
            temp_alert_color = ALERT_BACKGROUND_COLOR_BOTH
            alert_type = "HUMAN_VEHICLE"
        elif human_detected:
            temp_alert_message = "ALERT: HUMAN DETECTED!"
            temp_alert_color = ALERT_BACKGROUND_COLOR_HUMAN
            alert_type = "HUMAN"
        elif vehicle_detected:
            temp_alert_message = "ALERT: VEHICLE DETECTED!"
            temp_alert_color = ALERT_BACKGROUND_COLOR_VEHICLE
            alert_type = "VEHICLE"

        # Alert cooldown logic
        if temp_alert_message != current_alert_message:
            if current_alert_message:
                log_event(f"Alert cleared: {current_alert_message}")

            if temp_alert_message:
                if alert_type != last_alert_type or (current_time_sec - last_alert_time >= ALERT_COOLDOWN_SECONDS):
                    log_event(f"Alert triggered: {temp_alert_message}")
                    trigger_sound_alert(alert_type)
                    last_alert_time = current_time_sec
                    last_alert_type = alert_type
                else:
                    log_event(f"(Cooldown) Alert repeated too soon: {temp_alert_message}")

            current_alert_message = temp_alert_message
            alert_background_color = temp_alert_color
            alert_display_start_time = current_time_sec

        elif current_alert_message and (current_time_sec - alert_display_start_time >= ALERT_DISPLAY_DURATION_SECONDS):
            if not (fire_detected or human_detected or vehicle_detected):
                log_event(f"Alert cleared (timeout): {current_alert_message}")
                current_alert_message = ""
                alert_background_color = (0, 0, 0)

        # Draw alert banner
        if current_alert_message:
            text_size = cv2.getTextSize(current_alert_message, ALERT_TEXT_FONT, ALERT_TEXT_SCALE, ALERT_TEXT_THICKNESS)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = 70
            cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10),
                          (text_x + text_size[0] + 10, text_y + 10),
                          alert_background_color, -1)
            cv2.putText(frame, current_alert_message, (text_x, text_y),
                        ALERT_TEXT_FONT, ALERT_TEXT_SCALE, ALERT_TEXT_COLOR, ALERT_TEXT_THICKNESS)

        # FPS
        if fps_frame_count >= 10:
            now = time.time()
            fps = fps_frame_count / (now - fps_start_time)
            fps_start_time = now
            fps_frame_count = 0

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display mode
        mode_text = "Mode: IR" if IR_MODE_ENABLED else "Mode: Normal"
        rotation_text = f"Rotation: {CURRENT_ROTATION_ANGLE}°"
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, rotation_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show output
        cv2.imshow("Combined Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            log_event("ESC pressed. Exiting.")
            break
        elif key == ord('i'):
            IR_MODE_ENABLED = True
            log_event("IR mode enabled.")
        elif key == ord('e'):
            IR_MODE_ENABLED = False
            log_event("Normal mode enabled.")
        elif key == ord('r'):
            CURRENT_ROTATION_ANGLE = (CURRENT_ROTATION_ANGLE + 90) % 360
            log_event(f"Rotated to {CURRENT_ROTATION_ANGLE}°.")

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Detection completed.")

if __name__ == "__main__":
    run_combined_detection()
