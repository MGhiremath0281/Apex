from ultralytics import YOLO
import cv2
import time

# --- Parameters ---
CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5
IMG_SIZE = 640  # Lower for faster detection

# Zone size (bigger area to cover runway)
ZONE_TOP_LEFT_DEFAULT = (80, 100)
ZONE_BOTTOM_RIGHT_DEFAULT = (560, 440)

def put_text(img, text, pos, color=(0, 255, 0)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

def main():
    print("Loading YOLO model...")
    model = YOLO(r'C:\Users\adity\Desktop\OBJDETECT\yolov8l.pt')
    print("Model loaded.")

    ip_stream_url = "http://10.236.237.207:8080/video"  # <-- updated here
    cap = cv2.VideoCapture(ip_stream_url)

    if not cap.isOpened():
        print(f"Could not open stream at {ip_stream_url}")
        return

    zone_top_left = list(ZONE_TOP_LEFT_DEFAULT)
    zone_bottom_right = list(ZONE_BOTTOM_RIGHT_DEFAULT)

    # Setup window & callback
    cv2.namedWindow("IR Runway Clearance")

    fps_counter = []
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed. Reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(ip_stream_url)
            continue

        # Resize early for speed
        frame = cv2.resize(frame, (640, 480))

        # Fake IR effect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fake_ir = cv2.bitwise_not(gray)
        ir_frame = cv2.cvtColor(fake_ir, cv2.COLOR_GRAY2BGR)

        # YOLO detection
        results = model.predict(
            ir_frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False
        )
        annotated_frame = results[0].plot()

        # Check objects inside zone
        objects_in_zone = 0
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if zone_top_left[0] <= cx <= zone_bottom_right[0] and zone_top_left[1] <= cy <= zone_bottom_right[1]:
                objects_in_zone += 1

        runway_clear = (objects_in_zone == 0)

        # Zone color & text
        if runway_clear:
            zone_color = (0, 255, 0)
            text = "RUNWAY CLEAR"
        else:
            zone_color = (0, 0, 255)
            text = "RUNWAY NOT CLEAR"

        # --- Draw shaded region ---
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, tuple(zone_top_left), tuple(zone_bottom_right), zone_color, -1)
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

        # Draw border
        cv2.rectangle(annotated_frame, tuple(zone_top_left), tuple(zone_bottom_right), zone_color, 2)

        # Status text
        put_text(annotated_frame, text, (20, 40), zone_color)

        # FPS calculation
        current_time = time.time()
        fps = 1 / (current_time - last_time) if last_time else 0
        last_time = current_time
        put_text(annotated_frame, f"FPS: {fps:.1f}", (500, 30), (255, 255, 255))

        # Show
        cv2.imshow("IR Runway Clearance", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
