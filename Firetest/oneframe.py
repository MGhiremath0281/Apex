import cv2
from ultralytics import YOLO

# --- Model Loading ---
person_vehicle_model = YOLO('yolov8x.pt')
fire_model = YOLO('yolofire.pt')

# --- Configuration ---
input_source = r'c:\Users\adity\Desktop\OBJDETECT\data\firedata2.mp4'

# Reduce skip to slow it down a bit
SKIP_FRAMES = 2  # skip every 2nd frame only

def process_frame(frame):
    # Resize to smaller for faster processing
    frame = cv2.resize(frame, (640, 360))  # keep smaller for smoothness

    display_frame = frame.copy()

    # Use slightly lower image size for YOLO for faster speed
    results_general = person_vehicle_model(display_frame, verbose=False, imgsz=320)
    results_fire = fire_model(display_frame, verbose=False, imgsz=320)

    boxes_general = results_general[0].boxes
    boxes_fire = results_fire[0].boxes

    for box in boxes_general:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = person_vehicle_model.names[cls]

        if label not in ['fire', 'smoke', 'flame', 'blaze']:
            color = (0, 255, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for box in boxes_fire:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = fire_model.names[cls]

        color = (0, 0, 255)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return display_frame

if isinstance(input_source, (str, int)):
    cap = cv2.VideoCapture(input_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {input_source}.")
    else:
        print(f"Processing video/webcam: {input_source}")
        cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detections", 800, 600)  # Smaller screen size

        frame_counter = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or error reading frame.")
                break

            frame_counter += 1
            if frame_counter % SKIP_FRAMES != 0:
                continue

            processed_frame = process_frame(frame)

            cv2.imshow("Detections", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Video processing finished.")
else:
    print("Invalid input_source. Please specify an image path, video path, or webcam ID (0, 1, ...).")
