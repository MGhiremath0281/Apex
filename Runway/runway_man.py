from ultralytics import YOLO
import cv2
import time

# --- Constants ---
CONFIDENCE_THRESHOLD = 0.55
NMS_IOU_THRESHOLD = 0.5
IMG_SIZE = 960

# Fixed "landing zone" (change if needed)
ZONE_TOP_LEFT = (150, 150)
ZONE_BOTTOM_RIGHT = (500, 400)

# --- Helper function to draw text with shadow ---
def put_text_shadow(img, text, pos, font_scale, color, thickness):
    cv2.putText(img, text, (pos[0] + 2, pos[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def main():
    print("Loading YOLO model...")
    model = YOLO(r"C:\Users\adity\Desktop\OBJDETECT\yolov8l.pt")  # Change if needed
    print("YOLO model loaded successfully.")

    # --- Video file path ---
    video_path = r"C:\Users\adity\Desktop\OBJDETECT\data\LANDING AT MEWAR HELIPAD BY MEWAR HELICOPTERS - SHREE MEWAR HELICOPTER SERVICES (720p, h264).mp4"  # Change to your video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended or failed to read frame.")
            break

        # Resize for faster processing (optional)
        frame = cv2.resize(frame, (640, 480))

        # Convert to pure grayscale "IR" style
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ir_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)  # Convert back so YOLO accepts

        # Inference
        results = model.predict(
            ir_frame,
            device=0,
            conf=CONFIDENCE_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False
        )

        # Annotated frame from YOLO
        annotated_frame = results[0].plot()

        # Check objects in zone
        objects_in_zone = 0
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if (ZONE_TOP_LEFT[0] < cx < ZONE_BOTTOM_RIGHT[0]) and (ZONE_TOP_LEFT[1] < cy < ZONE_BOTTOM_RIGHT[1]):
                objects_in_zone += 1
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Runway status
        if objects_in_zone == 0:
            status_text = "RUNWAY CLEAR"
            zone_color = (0, 255, 0)
        else:
            status_text = "RUNWAY NOT CLEAR"
            zone_color = (0, 0, 255)

        # Draw landing zone rectangle
        cv2.rectangle(annotated_frame, ZONE_TOP_LEFT, ZONE_BOTTOM_RIGHT, zone_color, 3)

        # Draw status text
        put_text_shadow(annotated_frame, status_text, (20, 40), 1, zone_color, 2)

        # Show frame
        cv2.imshow("Runway Clearance (Black & White IR)", annotated_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Processing finished.")

if __name__ == "__main__":
    main()
