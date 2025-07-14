import cv2
from ultralytics import YOLO
import time
import numpy as np

# --- Configuration ---
VIDEO_PATH = r"C:\Users\adity\Desktop\OBJDETECT\data\birds.mp4"
YOLO_MODEL_PATH = r"C:\Users\adity\Desktop\OBJDETECT\yolov8x.pt"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
IMG_SIZE = 640

REGION_TOP_LEFT = (200, 150)
REGION_BOTTOM_RIGHT = (500, 400)

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)

# You can define a constant for box thickness here for easy adjustment
BBOX_THICKNESS = 4 # Increased thickness, e.g., from 2 to 4 or 5

# --- Helper Functions ---
def put_text_on_frame(img, text, pos, color=COLOR_WHITE, font_scale=0.7, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def draw_region_of_interest(frame, top_left, bottom_right, color, thickness=2, fill_alpha=0.2):
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)
    cv2.rectangle(frame, top_left, bottom_right, color, thickness)

def main_video():
    print("Loading YOLO model...")
    try:
        model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    print("Model loaded.")

    print(f"Opening video file: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{VIDEO_PATH}'. Please check the path and file integrity.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    last_frame_time = time.time()
    avg_fps = 0.0
    fps_alpha = 0.9

    print("Starting video processing...")
    cv2.namedWindow("Bird Detection - Video", cv2.WINDOW_NORMAL)

    detection_history = []
    HISTORY_LENGTH = 10
    DETECTION_THRESHOLD = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream reached or error reading frame.")
            break

        results_generator = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD,
                                  imgsz=IMG_SIZE, verbose=False, stream=True, device=0, half=True)

        annotated_frame = frame.copy()
        current_frame_bird_in_region = False
        detections_count = 0

        for result in results_generator:
            if result.boxes:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]

                    if cls_name == "bird":
                        detections_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        if (REGION_TOP_LEFT[0] <= center_x <= REGION_BOTTOM_RIGHT[0] and
                            REGION_TOP_LEFT[1] <= center_y <= REGION_BOTTOM_RIGHT[1]):
                            # Changed thickness here
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), COLOR_RED, BBOX_THICKNESS)
                            put_text_on_frame(annotated_frame, f"{cls_name} ({conf:.2f})",
                                              (x1, y1 - 10), COLOR_RED, font_scale=0.6)
                        else:
                            # Changed thickness here
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), COLOR_GREEN, BBOX_THICKNESS)
                            put_text_on_frame(annotated_frame, f"{cls_name} ({conf:.2f})",
                                              (x1, y1 - 10), COLOR_GREEN, font_scale=0.6)

        # --- Debouncing / Alert Logic ---
        detection_history.append(current_frame_bird_in_region)
        if len(detection_history) > HISTORY_LENGTH:
            detection_history.pop(0)

        num_detections_in_history = sum(detection_history)
        alert_triggered = num_detections_in_history >= DETECTION_THRESHOLD

        # --- Visual Feedback ---
        region_color = COLOR_BLUE
        status_message = "No Bird in Region"
        status_color = COLOR_GREEN

        if current_frame_bird_in_region:
            region_color = COLOR_YELLOW
            status_message = f"Bird detected (current: {detections_count})"
            status_color = COLOR_YELLOW

        if alert_triggered:
            region_color = COLOR_RED
            status_message = f"ALERT: BIRD IN REGION! ({num_detections_in_history}/{HISTORY_LENGTH})"
            status_color = COLOR_RED

        # The ROI thickness can also be adjusted, if desired
        draw_region_of_interest(annotated_frame, REGION_TOP_LEFT, REGION_BOTTOM_RIGHT, region_color, thickness=2, fill_alpha=0.1)
        put_text_on_frame(annotated_frame, status_message, (30, 50), status_color, font_scale=1.0, thickness=3)

        # Display FPS
        current_time = time.time()
        instant_fps = 1 / (current_time - last_frame_time) if (current_time - last_frame_time) > 0 else 0
        avg_fps = (fps_alpha * instant_fps) + ((1 - fps_alpha) * avg_fps)
        last_frame_time = current_time
        put_text_on_frame(annotated_frame, f"FPS: {avg_fps:.1f}", (frame_width - 150, 30), COLOR_WHITE)
        put_text_on_frame(annotated_frame, f"Detections: {detections_count}", (30, 90), COLOR_WHITE)

        cv2.imshow("Bird Detection - Video", annotated_frame)

        key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
        if key == ord("q"):
            print("Quitting program.")
            break
        elif key == ord("p"):
            cv2.waitKey(-1)
        elif key == ord("r"):
            print("Region reset (feature not implemented for static ROI).")

    cap.release()
    cv2.destroyAllWindows()
    print("Application terminated.")

if __name__ == "__main__":
    main_video()