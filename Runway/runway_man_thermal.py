from ultralytics import YOLO
import cv2
import time
import os
from collections import deque # For FPS smoothing

# --- Configuration Constants ---
CONFIDENCE_THRESHOLD = 0.55       # Minimum confidence for a detection
NMS_IOU_THRESHOLD = 0.5           # IoU threshold for Non-Maximum Suppression
IMG_SIZE = 960                    # Model input image size (affects speed/accuracy trade-off)

# Define relevant object classes that can block the landing zone (COCO dataset classes)
# You can customize this list based on what truly constitutes an "obstacle"
OBSTACLE_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'bench', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
    # Add other large items if relevant, e.g., 'dining table', 'couch', 'bed' if they can be outside
]

# Zone coordinates (adjust these as needed for your video/camera)
# Example values, you'll need to fine-tune for your specific video
ZONE_TOP_LEFT = (100, 150) # (x1, y1)
ZONE_BOTTOM_RIGHT = (540, 400) # (x2, y2)

# Stabilization parameters for status change
STABLE_FRAMES_REQUIRED = 10 # Number of consecutive frames to confirm status change
                             # Increased for more robust status transitions

# --- Helper functions ---
def put_text_shadow(img, text, pos, font_scale, color, thickness):
    # Shadow
    cv2.putText(img, text, (pos[0] + 3, pos[1] + 3),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Main text
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def check_significant_overlap(box_coords, zone_tl, zone_br, min_overlap_percentage=0.5):
    """
    Checks if a significant percentage of the detected object's bounding box
    is within the defined zone.
    box_coords: (x1, y1, x2, y2) of the detected object
    zone_tl: (zx1, zy1) top-left of the zone
    zone_br: (zx2, zy2) bottom-right of the zone
    min_overlap_percentage: Minimum percentage of the object's area that must overlap the zone.
    """
    bx1, by1, bx2, by2 = box_coords
    zx1, zy1 = zone_tl
    zx2, zy2 = zone_br

    # Calculate intersection rectangle coordinates
    ix1 = max(bx1, zx1)
    iy1 = max(by1, zy1)
    ix2 = min(bx2, zx2)
    iy2 = min(by2, zy2)

    # Calculate intersection area
    intersection_width = max(0, ix2 - ix1)
    intersection_height = max(0, iy2 - iy1)
    intersection_area = intersection_width * intersection_height

    # Calculate object bounding box area
    object_width = bx2 - bx1
    object_height = by2 - by1
    object_area = object_width * object_height

    if object_area == 0: # Avoid division by zero
        return False

    overlap_ratio = intersection_area / object_area
    return overlap_ratio >= min_overlap_percentage

def main():
    print("Loading YOLO model...")
    # Using yolov8x.pt as specified in your code
    model = YOLO(r"C:\Users\adity\Desktop\OBJDETECT\yolov8x.pt")
    print("YOLO model loaded successfully.")

    video_path = r"C:\Users\adity\Desktop\OBJDETECT\data\Recording 2025-07-13 004951.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    # Get video frame dimensions for resizing consistent display
    # We will resize the video frame to 640x480 for display consistency,
    # but the model will infer on IMG_SIZE.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    display_size = (640, 480) # Fixed display size for consistent UI

    # Stabilization counters
    current_clear_streak = 0
    current_blocked_streak = 0
    final_status = "LANDING ZONE CLEAR" # Initial status
    status_color = (0, 255, 0) # Green for clear

    # For FPS calculation
    frame_times = deque(maxlen=10) # Smoother FPS over 10 frames

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended or error reading frame. Looping video for demonstration.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
            continue

        # Resize frame for consistent display window size
        display_frame = cv2.resize(frame, display_size)
        
        # --- Simulate EO/IR view for DISPLAY ONLY ---
        # The model will still process the original BGR frame for better accuracy
        # if the model was trained on color images.
        ir_display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
        ir_display_frame = cv2.cvtColor(ir_display_frame, cv2.COLOR_GRAY2BGR)
        # Optional: apply a color tint for more 'IR' feel
        ir_display_frame = cv2.applyColorMap(cv2.convertScaleAbs(ir_display_frame, alpha=1.5), cv2.COLORMAP_JET)
        
        # --- Model Inference ---
        # Crucial: Feed the ORIGINAL frame (resized to model's input size) for detection
        # This prevents loss of color information if yolov8x.pt was trained on color.
        results = model.predict(
            frame, # Use original frame for prediction (or resized frame for a faster pipeline)
                   # If performance is an issue, consider `cv2.resize(frame, (IMG_SIZE, IMG_SIZE))` before predict.
            device=0,
            conf=CONFIDENCE_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            imgsz=IMG_SIZE, # Model will automatically resize the input frame to this
            verbose=False,
            half=True # Use half-precision for faster inference on compatible GPUs
        )

        # Plot detections on the IR display frame
        # We need to manually plot on `ir_display_frame` because `results[0].plot()`
        # operates on the image it was given during prediction.
        # So, we'll draw based on the resized `display_frame` coordinates.
        objects_in_zone_current_frame = 0
        detected_obstacle_names = []

        # Resize detected boxes to match `display_size` for drawing
        # YOLO's results.boxes.xyxy are relative to the *input* image size (IMG_SIZE).
        # We need to scale them to the `display_size` (640x480).
        scale_x = display_size[0] / IMG_SIZE
        scale_y = display_size[1] / IMG_SIZE

        for box in results[0].boxes:
            class_id = int(box.cls)
            class_name = results[0].names[class_id]
            
            # Check if this class is a relevant obstacle
            if class_name in OBSTACLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Scale coordinates for drawing on `ir_display_frame`
                dx1, dy1 = int(x1 * scale_x), int(y1 * scale_y)
                dx2, dy2 = int(x2 * scale_x), int(y2 * scale_y)

                # Check if this obstacle significantly overlaps with the zone
                if check_significant_overlap((dx1, dy1, dx2, dy2), ZONE_TOP_LEFT, ZONE_BOTTOM_RIGHT):
                    objects_in_zone_current_frame += 1
                    detected_obstacle_names.append(class_name.upper())
                    # Draw a red rectangle for objects *inside* the zone
                    cv2.rectangle(ir_display_frame, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
                    put_text_shadow(ir_display_frame, class_name, (dx1, dy1 - 10), 0.6, (0, 0, 255), 1)
                else:
                    # Draw a green rectangle for obstacles *outside* the zone (still detected)
                    cv2.rectangle(ir_display_frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                    put_text_shadow(ir_display_frame, class_name, (dx1, dy1 - 10), 0.6, (0, 255, 0), 1)


        # --- Status Update Logic (Debouncing) ---
        if objects_in_zone_current_frame == 0:
            current_clear_streak += 1
            current_blocked_streak = 0 # Reset blocked streak
            if current_clear_streak >= STABLE_FRAMES_REQUIRED:
                final_status = "LANDING ZONE CLEAR"
                status_color = (0, 255, 0) # Green
        else:
            current_blocked_streak += 1
            current_clear_streak = 0 # Reset clear streak
            if current_blocked_streak >= STABLE_FRAMES_REQUIRED:
                final_status = "LANDING ZONE BLOCKED"
                status_color = (0, 0, 255) # Red

        # --- Visualizations ---
        # Draw the zone rectangle on the display frame
        cv2.rectangle(ir_display_frame, ZONE_TOP_LEFT, ZONE_BOTTOM_RIGHT, status_color, 3)

        # Optional semi-transparent overlay within the zone
        overlay_zone = ir_display_frame.copy()
        cv2.rectangle(overlay_zone, ZONE_TOP_LEFT, ZONE_BOTTOM_RIGHT, status_color, -1)
        alpha = 0.15
        cv2.addWeighted(overlay_zone, alpha, ir_display_frame, 1 - alpha, 0, ir_display_frame)

        # Main status text
        put_text_shadow(ir_display_frame, final_status, (20, 40), 1, status_color, 2)

        # Display detected obstacles in zone if any
        if detected_obstacle_names and final_status == "LANDING ZONE BLOCKED":
            unique_obstacles = sorted(list(set(detected_obstacle_names)))
            obstacle_text = "Obstacles: " + ", ".join(unique_obstacles)
            put_text_shadow(ir_display_frame, obstacle_text, (20, 70), 0.7, (0, 0, 255), 1)
            
            # Additional alert text for BLOCKED status
            if final_status == "LANDING ZONE BLOCKED":
                 put_text_shadow(ir_display_frame, "!!! ALERT !!!", (20, 100), 0.8, (0, 0, 255), 1)


        # --- FPS Display ---
        curr_time = time.time()
        frame_times.append(curr_time)
        if len(frame_times) > 1:
            avg_frame_time = (frame_times[-1] - frame_times[0]) / (len(frame_times) - 1)
            fps = 1 / avg_frame_time if avg_frame_time > 0 else 0
        else:
            fps = 0
        
        put_text_shadow(ir_display_frame, f"FPS: {fps:.1f}", (display_size[0] - 120, 25), 0.7, (255, 255, 255), 1)
        
        # Display current time
        current_datetime_str = time.strftime("%H:%M:%S", time.localtime())
        put_text_shadow(ir_display_frame, current_datetime_str, (display_size[0] - 130, 45), 0.6, (255, 255, 255), 1)


        cv2.imshow("Landing Zone Monitor", ir_display_frame)

        key = cv2.waitKey(30) & 0xFF # Adjust waitKey for video playback speed (30ms for ~30 FPS video)
        if key == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Finished.")

if __name__ == "__main__":
    main()