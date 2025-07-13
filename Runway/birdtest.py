from ultralytics import YOLO
import cv2
import time
from collections import deque

# --- Configuration ---
CONFIDENCE_THRESHOLD = 0.20
NMS_IOU_THRESHOLD = 0.7
IMG_SIZE = 1600

BIRD_CLASSES = ['bird']  # COCO class name for bird

ZONE_TOP_LEFT = (100, 100)
ZONE_BOTTOM_RIGHT = (540, 400)
STABLE_FRAMES_REQUIRED = 5

def put_text_shadow(img, text, pos, font_scale, color, thickness):
    cv2.putText(img, text, (pos[0] + 2, pos[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def check_overlap(box_coords, zone_tl, zone_br, min_overlap_percentage=0.4):
    bx1, by1, bx2, by2 = box_coords
    zx1, zy1 = zone_tl
    zx2, zy2 = zone_br

    ix1 = max(bx1, zx1)
    iy1 = max(by1, zy1)
    ix2 = min(bx2, zx2)
    iy2 = min(by2, zy2)

    intersection_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    object_area = (bx2 - bx1) * (by2 - by1)
    if object_area == 0:
        return False

    overlap_ratio = intersection_area / object_area
    return overlap_ratio >= min_overlap_percentage

def main():
    print("Loading YOLO model...")
    model = YOLO(r"C:\Users\adity\Desktop\OBJDETECT\yolov8x.pt")
    print("YOLO model loaded.")

    video_path = r"C:\Users\adity\Desktop\OBJDETECT\data\birds.mp4"  # Example: your bird scenario video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video at {video_path}")
        return

    display_size = (640, 480)
    clear_streak = 0
    blocked_streak = 0
    status = "LANDING ZONE CLEAR"
    status_color = (0, 255, 0)

    frame_times = deque(maxlen=20)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended or error. Restarting video.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        orig_h, orig_w = frame.shape[:2]
        display_frame = cv2.resize(frame, display_size)

        results = model.predict(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            device=0,
            verbose=False,
            half=False
        )

        birds_in_zone = 0

        scale_x = display_size[0] / orig_w
        scale_y = display_size[1] / orig_h

        for box in results[0].boxes:
            cls_id = int(box.cls)
            cls_name = results[0].names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            dx1, dy1 = int(x1 * scale_x), int(y1 * scale_y)
            dx2, dy2 = int(x2 * scale_x), int(y2 * scale_y)

            if cls_name == 'bird' and check_overlap((dx1, dy1, dx2, dy2), ZONE_TOP_LEFT, ZONE_BOTTOM_RIGHT):
                birds_in_zone += 1
                box_color = (0, 0, 255)
            else:
                box_color = (0, 255, 0)

            cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), box_color, 2)
            put_text_shadow(display_frame, cls_name, (dx1, dy1 - 8), 0.5, box_color, 1)

        if birds_in_zone == 0:
            clear_streak += 1
            blocked_streak = 0
            if clear_streak >= STABLE_FRAMES_REQUIRED:
                status = "LANDING ZONE CLEAR"
                status_color = (0, 255, 0)
        else:
            blocked_streak += 1
            clear_streak = 0
            if blocked_streak >= STABLE_FRAMES_REQUIRED:
                status = "BIRD ALERT: ZONE BLOCKED"
                status_color = (0, 0, 255)

        cv2.rectangle(display_frame, ZONE_TOP_LEFT, ZONE_BOTTOM_RIGHT, status_color, 3)
        overlay = display_frame.copy()
        cv2.rectangle(overlay, ZONE_TOP_LEFT, ZONE_BOTTOM_RIGHT, status_color, -1)
        cv2.addWeighted(overlay, 0.15, display_frame, 0.85, 0, display_frame)

        put_text_shadow(display_frame, status, (20, 40), 1, status_color, 2)

        if birds_in_zone > 0:
            put_text_shadow(display_frame, "!!! ALERT: BIRD DETECTED !!!", (20, 80), 0.8, (0, 0, 255), 2)

        curr_time = time.time()
        frame_times.append(curr_time)
        if len(frame_times) > 1:
            avg_time = (frame_times[-1] - frame_times[0]) / (len(frame_times) - 1)
            fps = 1 / avg_time if avg_time > 0 else 0
        else:
            fps = 0

        put_text_shadow(display_frame, f"FPS: {fps:.1f}", (display_size[0] - 120, 25), 0.7, (255, 255, 255), 1)
        time_str = time.strftime("%H:%M:%S", time.localtime())
        put_text_shadow(display_frame, time_str, (display_size[0] - 130, 45), 0.6, (255, 255, 255), 1)

        cv2.imshow("Drone Bird Detection", display_frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Finished.")

if __name__ == "__main__":
    main()
