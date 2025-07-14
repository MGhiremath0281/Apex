import cv2
import torch
from ultralytics import YOLO

# --- Device Selection ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- Model Loading ---
person_vehicle_model = YOLO('yolov8x.pt').to(device)
fire_model = YOLO('yolofirenew.pt').to(device)

# --- Configuration ---
input_source = r'c:\Users\adity\Desktop\OBJDETECT\data\firedata2.mp4'
SKIP_FRAMES = 2  # Skip every 2nd frame for speed

def process_frame(frame):
    # Resize frame for faster inference
    frame = cv2.resize(frame, (640, 360))
    display_frame = frame.copy()

    # YOLO Inference
    results_general = person_vehicle_model.predict(display_frame, imgsz=320, verbose=False, device=device)
    results_fire = fire_model.predict(display_frame, imgsz=320, verbose=False, device=device)

    boxes_general = results_general[0].boxes
    boxes_fire = results_fire[0].boxes

    # Draw general model detections
    for box in boxes_general:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = person_vehicle_model.names.get(cls, f"id:{cls}")

        # Avoid overlapping with fire labels
        if label.lower() not in ['fire', 'smoke', 'flame', 'blaze']:
            color = (0, 255, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw fire model detections
    for box in boxes_fire:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = fire_model.names.get(cls, f"id:{cls}")

        color = (0, 0, 255)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return display_frame

# --- Main Video Loop ---
if isinstance(input_source, (str, int)):
    cap = cv2.VideoCapture(input_source)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video source: {input_source}")
    else:
        print(f"🎥 Processing: {input_source}")
        cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detections", 800, 600)

        frame_counter = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✅ End of stream.")
                break

            frame_counter += 1
            if frame_counter % SKIP_FRAMES != 0:
                continue

            try:
                processed = process_frame(frame)
                cv2.imshow("Detections", processed)
            except Exception as e:
                print(f"⚠️ Error during frame processing: {e}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⏹️ Stopped by user.")
                break

        cap.release()
        cv2.destroyAllWindows()
        print("✅ Video processing complete.")
else:
    print("❌ Invalid input_source. Use a valid image/video path or webcam index.")
