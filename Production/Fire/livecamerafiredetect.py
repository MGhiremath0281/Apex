import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO

# --- CONFIGURATION ---
BASE_YOLO_MODEL_PATH = r'c:\Users\adity\Desktop\OBJDETECT\yolov8x.pt'
FIRE_YOLO_MODEL_PATH = r'c:\Users\adity\Desktop\OBJDETECT\yolofirenew.pt'

CAMERA_SOURCE = 0  # 0 = default webcam; or "rtsp://..." for IP cam

HUMAN_CLASS_NAME = 'person'
VEHICLE_CLASS_NAMES = ['car', 'truck', 'bus', 'motorcycle']
FIRE_CLASS_NAME = 'Fire'

CONFIDENCE_THRESHOLD_BASE = 0.5
CONFIDENCE_THRESHOLD_FIRE = 0.3

ALERT_TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
ALERT_TEXT_SCALE = 1.5
ALERT_TEXT_THICKNESS = 3
ALERT_TEXT_COLOR = (255, 255, 255)
ALERT_BG_COLOR_FIRE = (0, 0, 255)
ALERT_BG_COLOR_HUMAN = (255, 0, 0)
ALERT_BG_COLOR_VEHICLE = (0, 255, 255)

LOGS_DIRECTORY = 'detection_logs_live'
LOG_FILE_NAME_PREFIX = 'live_detection_log_'

# --- GLOBALS ---
LOG_FILE = None
CURRENT_FRAME_NUMBER = 0
last_print_time = time.time()

# --- FUNCTION: class ID lookup ---
def get_class_id(model, class_name):
    for class_id, name in model.names.items():
        if name.strip().lower() == class_name.strip().lower():
            return class_id
    return None

# --- SETUP LOGGING ---
def setup_logging():
    global LOG_FILE
    if not os.path.exists(LOGS_DIRECTORY):
        os.makedirs(LOGS_DIRECTORY)
        print(f"Created log directory: {LOGS_DIRECTORY}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(LOGS_DIRECTORY, f"{LOG_FILE_NAME_PREFIX}{timestamp}.txt")
    try:
        LOG_FILE = open(log_file_path, 'a')
        LOG_FILE.write(f"--- Live Detection Log Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        LOG_FILE.write(f"Camera Source: {CAMERA_SOURCE}\n")
        LOG_FILE.write(f"Base Model: {os.path.basename(BASE_YOLO_MODEL_PATH)}\n")
        LOG_FILE.write(f"Fire Model: {os.path.basename(FIRE_YOLO_MODEL_PATH)}\n")
        LOG_FILE.write("-"*60 + "\n")
        print(f"Logging to: {log_file_path}")
    except Exception as e:
        print(f"Failed to open log file: {e}")
        LOG_FILE = None

def log_event(msg):
    global LOG_FILE, CURRENT_FRAME_NUMBER
    if LOG_FILE:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        LOG_FILE.write(f"[{ts} | Frame {CURRENT_FRAME_NUMBER}] {msg}\n")
        LOG_FILE.flush()

# --- MAIN FUNCTION ---
def run_live_detection():
    global CURRENT_FRAME_NUMBER, last_print_time

    setup_logging()
    if not LOG_FILE:
        print("Logging disabled due to file error.")

    # Load YOLO models
    try:
        base_model = YOLO(BASE_YOLO_MODEL_PATH)
        fire_model = YOLO(FIRE_YOLO_MODEL_PATH)
        print("✅ Models loaded.")
    except Exception as e:
        print(f"Model load error: {e}")
        return

    HUMAN_ID = get_class_id(base_model, HUMAN_CLASS_NAME)
    VEHICLE_IDS = [get_class_id(base_model, vn) for vn in VEHICLE_CLASS_NAMES if get_class_id(base_model, vn) is not None]
    FIRE_ID = get_class_id(fire_model, FIRE_CLASS_NAME)

    if FIRE_ID is None:
        print(f"ERROR: Fire class '{FIRE_CLASS_NAME}' not found in fire model.")
        if LOG_FILE:
            LOG_FILE.close()
        return

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print("Failed to open camera.")
        if LOG_FILE:
            LOG_FILE.close()
        return

    fps_start = time.time()
    fps_count = 0
    fps = 0.0

    alert_msg = ""
    alert_bg = (0,0,0)
    alert_start = 0
    ALERT_DURATION = 3  # seconds

    print("Starting live detection... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        CURRENT_FRAME_NUMBER += 1
        fps_count += 1

        detections = []

        cls_list = [HUMAN_ID] if HUMAN_ID is not None else []
        cls_list += VEHICLE_IDS

        if cls_list:
            res = base_model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD_BASE, classes=cls_list)
            for r in res:
                for b in r.boxes:
                    x1,y1,x2,y2 = map(int, b.xyxy[0])
                    conf = float(b.conf[0])
                    cid = int(b.cls[0])
                    detections.append(('base',cid,conf,(x1,y1,x2,y2)))

        resf = fire_model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD_FIRE, classes=FIRE_ID)
        for r in resf:
            for b in r.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0])
                cid = int(b.cls[0])
                detections.append(('fire',cid,conf,(x1,y1,x2,y2)))

        fire_flag = human_flag = vehicle_flag = False

        for model_src, cid, conf, bbox in detections:
            x1,y1,x2,y2 = bbox
            label, color = "", (0,0,0)
            if model_src == 'fire' and cid == FIRE_ID:
                fire_flag = True
                label = f"FIRE {conf:.2f}"
                color = ALERT_BG_COLOR_FIRE
            elif model_src == 'base':
                if cid == HUMAN_ID:
                    human_flag = True
                    label = f"Human {conf:.2f}"
                    color = ALERT_BG_COLOR_HUMAN
                elif cid in VEHICLE_IDS:
                    vehicle_flag = True
                    vn = base_model.names[cid].capitalize()
                    label = f"{vn} {conf:.2f}"
                    color = ALERT_BG_COLOR_VEHICLE

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        current_time = time.time()
        if current_time - last_print_time >= 3:
            if fire_flag:
                print("🔥 FIRE DETECTED!")
            if human_flag and vehicle_flag:
                print("👤 Human and 🚗 Vehicle detected!")
            elif human_flag:
                print("👤 Human detected!")
            elif vehicle_flag:
                print("🚗 Vehicle detected!")
            last_print_time = current_time

        if fire_flag:
            new_alert, new_bg = "ALERT: FIRE DETECTED!", ALERT_BG_COLOR_FIRE
        elif human_flag and vehicle_flag:
            new_alert, new_bg = "ALERT: HUMAN & VEHICLE DETECTED!", ALERT_BG_COLOR_VEHICLE
        elif human_flag:
            new_alert, new_bg = "ALERT: HUMAN DETECTED!", ALERT_BG_COLOR_HUMAN
        elif vehicle_flag:
            new_alert, new_bg = "ALERT: VEHICLE DETECTED!", ALERT_BG_COLOR_VEHICLE
        else:
            new_alert, new_bg = "", (0,0,0)

        if new_alert:
            alert_msg, alert_bg = new_alert, new_bg
            alert_start = time.time()

        if alert_msg and (time.time() - alert_start <= ALERT_DURATION):
            size = cv2.getTextSize(alert_msg, ALERT_TEXT_FONT,
                                   ALERT_TEXT_SCALE, ALERT_TEXT_THICKNESS)[0]
            tx = (frame.shape[1] - size[0]) // 2
            ty = 70
            cv2.rectangle(frame, (tx-10, ty-size[1]-10),
                          (tx+size[0]+10, ty+10), alert_bg, -1)
            cv2.putText(frame, alert_msg, (tx, ty),
                        ALERT_TEXT_FONT, ALERT_TEXT_SCALE, ALERT_TEXT_COLOR, ALERT_TEXT_THICKNESS)

        if time.time() - fps_start >= 1:
            fps = fps_count / (time.time() - fps_start)
            fps_start = time.time()
            fps_count = 0
        cv2.putText(frame, f"FPS: {fps:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

        cv2.imshow("Live Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if LOG_FILE:
        LOG_FILE.write("-"*60+"\n")
        LOG_FILE.write(f"--- Detection Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        LOG_FILE.close()
    print("✅ System shut down.")

# --- RUN ---
if __name__ == "__main__":
    run_live_detection()
