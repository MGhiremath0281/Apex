from ultralytics import YOLO
import cv2

# --- Load YOLO model ---
model = YOLO(r'C:\Users\adity\Desktop\OBJDETECT\yolov8l.pt')

# --- Zone coordinates ---
ZONE_TOP_LEFT = (150, 150)
ZONE_BOTTOM_RIGHT = (500, 400)

def put_text(img, text, pos, color=(0, 255, 0)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

def process_frame(frame):
    # Convert to grayscale and invert (IR-like)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fake_ir = cv2.bitwise_not(gray)
    fake_ir_3ch = cv2.cvtColor(fake_ir, cv2.COLOR_GRAY2BGR)

    # Predict with YOLO
    results = model.predict(fake_ir_3ch, conf=0.55, iou=0.5, imgsz=960, device=0, verbose=False)
    annotated_frame = results[0].plot()

    objects_in_zone = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        if ZONE_TOP_LEFT[0] < cx < ZONE_BOTTOM_RIGHT[0] and ZONE_TOP_LEFT[1] < cy < ZONE_BOTTOM_RIGHT[1]:
            objects_in_zone.append(1)

    runway_clear = len(objects_in_zone) == 0

    if runway_clear:
        zone_color = (0, 255, 0)
        text = "RUNWAY CLEAR"
        text_color = (0, 255, 0)
    else:
        zone_color = (0, 0, 255)
        text = "RUNWAY NOT CLEAR"
        text_color = (0, 0, 255)

    # Draw zone rectangle and status text
    cv2.rectangle(annotated_frame, ZONE_TOP_LEFT, ZONE_BOTTOM_RIGHT, zone_color, 3)
    put_text(annotated_frame, text, (30, 50), text_color)

    return annotated_frame

# --- Choose camera index ---
camera_index = 0  # Try 0, or 1, or 2 depending on your phone webcam

# --- Open webcam ---
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"Could not open camera index {camera_index}. Check camera connection or app.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out_frame = process_frame(frame)
    cv2.imshow("IR Runway", out_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
