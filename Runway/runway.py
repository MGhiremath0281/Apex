from ultralytics import YOLO
import cv2

# Parameters
CONFIDENCE_THRESHOLD = 0.55
NMS_IOU_THRESHOLD = 0.5
IMG_SIZE = 960

ZONE_TOP_LEFT_DEFAULT = (150, 150)
ZONE_BOTTOM_RIGHT_DEFAULT = (500, 400)

def put_text(img, text, pos, color=(0, 255, 0)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

def main():
    # Load YOLO model
    model = YOLO(r'C:\Users\adity\Desktop\OBJDETECT\yolov8l.pt')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    zone_top_left = list(ZONE_TOP_LEFT_DEFAULT)
    zone_bottom_right = list(ZONE_BOTTOM_RIGHT_DEFAULT)

    is_dragging = False
    drag_offset_x, drag_offset_y = 0, 0

    def mouse_callback(event, x, y, flags, param):
        nonlocal is_dragging, drag_offset_x, drag_offset_y, zone_top_left, zone_bottom_right

        if event == cv2.EVENT_LBUTTONDOWN:
            if zone_top_left[0] <= x <= zone_bottom_right[0] and zone_top_left[1] <= y <= zone_bottom_right[1]:
                is_dragging = True
                drag_offset_x = x - zone_top_left[0]
                drag_offset_y = y - zone_top_left[1]
        elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
            new_x1 = x - drag_offset_x
            new_y1 = y - drag_offset_y

            width = zone_bottom_right[0] - zone_top_left[0]
            height = zone_bottom_right[1] - zone_top_left[1]

            zone_top_left[0] = new_x1
            zone_top_left[1] = new_y1
            zone_bottom_right[0] = new_x1 + width
            zone_bottom_right[1] = new_y1 + height
        elif event == cv2.EVENT_LBUTTONUP:
            is_dragging = False

    cv2.namedWindow("Runway Clearance")
    cv2.setMouseCallback("Runway Clearance", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        try:
            results = model.predict(
                frame,
                device=0,
                verbose=False,
                conf=CONFIDENCE_THRESHOLD,
                iou=NMS_IOU_THRESHOLD,
                imgsz=IMG_SIZE
            )
            annotated_frame = results[0].plot()
        except Exception as e:
            print(f"Error during prediction: {e}")
            break

        objects_in_zone = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if zone_top_left[0] < cx < zone_bottom_right[0] and zone_top_left[1] < cy < zone_bottom_right[1]:
                objects_in_zone.append(1)

        runway_clear = (len(objects_in_zone) == 0)

        if runway_clear:
            zone_color = (0, 255, 0)
            text = "RUNWAY CLEAR"
            text_color = (0, 255, 0)
        else:
            zone_color = (0, 0, 255)
            text = "RUNWAY NOT CLEAR"
            text_color = (0, 0, 255)

        cv2.rectangle(annotated_frame, tuple(zone_top_left), tuple(zone_bottom_right), zone_color, 3)
        put_text(annotated_frame, text, (30, 50), text_color)

        cv2.imshow("Runway Clearance", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
