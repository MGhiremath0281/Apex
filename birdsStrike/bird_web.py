import cv2
from ultralytics import YOLO
import time
import numpy as np

# --- Configuration ---
# Change to your webcam index (0, 1, etc.) or IP stream URL (e.g., "http://192.168.1.100:8080/video")
CAMERA_SOURCE = 0
YOLO_MODEL_PATH = r"C:\Users\adity\Desktop\OBJDETECT\yolov8x.pt" # Consider yolov8s/m for better real-time performance
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5 # NMS IOU threshold
IMG_SIZE = 640 # Default image size for YOLO inference. Adjust for performance vs. accuracy.

# Region of Interest (ROI) coordinates
REGION_TOP_LEFT = (200, 150)
REGION_BOTTOM_RIGHT = (500, 400)

# Colors (B, G, R)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)

# Bounding box and text rendering
BBOX_THICKNESS = 3 # Thickness of the bounding boxes (e.g., 2, 3, 4)
TEXT_FONT_SCALE = 0.6
TEXT_THICKNESS = 2

# --- Helper Functions ---
def put_text_on_frame(img, text, pos, color=COLOR_WHITE, font_scale=TEXT_FONT_SCALE, thickness=TEXT_THICKNESS):
    """Helper to draw text on an image."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def draw_region_of_interest(frame, top_left, bottom_right, color, thickness=2, fill_alpha=0.1):
    """Draws and optionally shades the region of interest."""
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1) # Filled rectangle
    cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)
    cv2.rectangle(frame, top_left, bottom_right, color, thickness) # Border

def main_camera():
    """Main function to run the bird detection on a live camera stream."""
    print("Loading YOLO model...")
    try:
        model = YOLO(YOLO_MODEL_PATH)
        # Optional: Export to TensorRT/ONNX for faster inference if you have a GPU
        # model.export(format='engine', half=True, device=0) # For TensorRT
        # model = YOLO(YOLO_MODEL_PATH.replace('.pt', '.engine')) # Load the exported engine
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    print("Model loaded.")

    print(f"Opening camera source: {CAMERA_SOURCE}")
    cap = cv2.VideoCapture(CAMERA_SOURCE)

    # Attempt to set camera resolution (may not be supported by all cameras/drivers)
    # Adjust these values based on your camera's capabilities and desired performance
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"Error: Could not open camera source '{CAMERA_SOURCE}'. "
              "Please check if the camera is connected and not in use by another application.")
        return

    # Get actual frame dimensions after opening (or setting)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_camera = cap.get(cv2.CAP_PROP_FPS) or 30 # Get camera FPS or default to 30

    # Initialize FPS counter for display
    last_frame_time = time.time()
    avg_fps = 0.0
    fps_alpha = 0.9 # Smoothing factor for FPS calculation

    print("Starting camera stream processing...")
    cv2.namedWindow("Bird Detection - Camera", cv2.WINDOW_NORMAL) # Allows window resizing

    # Debouncing / Alert Logic
    detection_history = []
    HISTORY_LENGTH = 15 # Number of frames to consider for debouncing (increased for live stream)
    DETECTION_THRESHOLD = 8 # Number of detections in history to trigger alert

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Attempting to reconnect...")
            cap.release()
            time.sleep(2) # Wait a bit before retrying
            cap = cv2.VideoCapture(CAMERA_SOURCE)
            if not cap.isOpened():
                print("Failed to reconnect to camera. Exiting.")
                break
            continue # Skip to next loop iteration if reconnection was needed

        # Resize frame before inference if your camera resolution is very high
        # This helps manage memory and speeds up detection if IMG_SIZE is smaller than frame
        if frame.shape[1] != IMG_SIZE or frame.shape[0] != int(IMG_SIZE * (frame.shape[0]/frame.shape[1])):
             # Maintain aspect ratio if resizing for YOLO input, otherwise just resize
             frame_resized = cv2.resize(frame, (IMG_SIZE, int(IMG_SIZE * (frame.shape[0]/frame.shape[1]))))
        else:
             frame_resized = frame

        # Perform YOLO inference
        # Use 'stream=True' for faster inference on successive frames
        # Use 'half=True' for FP16 inference on GPU
        # Use 'device=0' to explicitly use GPU 0, or 'cpu'
        results_generator = model(frame_resized, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD,
                                  imgsz=IMG_SIZE, verbose=False, stream=True, device=0, half=True)

        annotated_frame = frame.copy() # Draw on the original size frame
        current_frame_bird_in_region = False
        detections_count = 0

        # Iterate over the results generator
        for result in results_generator:
            if result.boxes: # Check if any bounding boxes exist in this result
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]

                    # Only consider 'bird' detections
                    if cls_name == "bird":
                        detections_count += 1
                        # Coordinates are for the resized frame; scale back to original frame for drawing
                        # Or, draw on frame_resized and then resize annotated_frame for display
                        # For simplicity, let's assume annotated_frame is original and scale box coords
                        
                        # Scale factor based on original frame vs. YOLO input frame
                        scale_x = frame.shape[1] / frame_resized.shape[1]
                        scale_y = frame.shape[0] / frame_resized.shape[0]

                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        
                        # Scale coordinates back to original frame size for drawing accuracy
                        x1_orig, y1_orig = int(x1 * scale_x), int(y1 * scale_y)
                        x2_orig, y2_orig = int(x2 * scale_x), int(y2 * scale_y)
                        
                        center_x = (x1_orig + x2_orig) // 2
                        center_y = (y1_orig + y2_orig) // 2

                        # Check if the bird's center is within the defined region
                        # IMPORTANT: Ensure REGION_TOP_LEFT/BOTTOM_RIGHT are defined relative to the ORIGINAL frame size
                        if (REGION_TOP_LEFT[0] <= center_x <= REGION_BOTTOM_RIGHT[0] and
                            REGION_TOP_LEFT[1] <= center_y <= REGION_BOTTOM_RIGHT[1]):
                            current_frame_bird_in_region = True
                            # Draw bounding box for birds in region with a distinct color
                            cv2.rectangle(annotated_frame, (x1_orig, y1_orig), (x2_orig, y2_orig), COLOR_RED, BBOX_THICKNESS)
                            put_text_on_frame(annotated_frame, f"{cls_name} ({conf:.2f})",
                                              (x1_orig, y1_orig - 10), COLOR_RED)
                        else:
                            # Draw bounding box for birds outside region
                            cv2.rectangle(annotated_frame, (x1_orig, y1_orig), (x2_orig, y2_orig), COLOR_GREEN, BBOX_THICKNESS)
                            put_text_on_frame(annotated_frame, f"{cls_name} ({conf:.2f})",
                                              (x1_orig, y1_orig - 10), COLOR_GREEN)

        # --- Debouncing / Alert Logic ---
        detection_history.append(current_frame_bird_in_region)
        if len(detection_history) > HISTORY_LENGTH:
            detection_history.pop(0) # Keep history length constant

        # Check if birds have been consistently detected in the region
        num_detections_in_history = sum(detection_history)
        alert_triggered = num_detections_in_history >= DETECTION_THRESHOLD

        # --- Visual Feedback ---
        region_color = COLOR_BLUE # Default color for ROI
        status_message = "No Bird in Region"
        status_color = COLOR_GREEN

        if current_frame_bird_in_region:
            region_color = COLOR_YELLOW # Immediate feedback
            status_message = f"Bird detected (current: {detections_count})"
            status_color = COLOR_YELLOW

        if alert_triggered:
            region_color = COLOR_RED # Persistent alert color
            status_message = f"ALERT: BIRD IN REGION! ({num_detections_in_history}/{HISTORY_LENGTH})"
            status_color = COLOR_RED
            # Optional: Add sound alerts or log to a file here
            # For Windows: import winsound; winsound.Beep(1000, 200) # Beep at 1000 Hz for 200 ms

        draw_region_of_interest(annotated_frame, REGION_TOP_LEFT, REGION_BOTTOM_RIGHT, region_color, thickness=2, fill_alpha=0.1)
        put_text_on_frame(annotated_frame, status_message, (30, 50), status_color, font_scale=1.0, thickness=3)

        # Display FPS
        current_time = time.time()
        instant_fps = 1 / (current_time - last_frame_time) if (current_time - last_frame_time) > 0 else 0
        avg_fps = (fps_alpha * instant_fps) + ((1 - fps_alpha) * avg_fps)
        last_frame_time = current_time
        put_text_on_frame(annotated_frame, f"FPS: {avg_fps:.1f}", (frame_width - 150, 30), COLOR_WHITE)
        put_text_on_frame(annotated_frame, f"Detections: {detections_count}", (30, 90), COLOR_WHITE)

        # Show the processed frame
        cv2.imshow("Bird Detection - Camera", annotated_frame)

        # Handle keyboard input (waitKey(1) is for live stream, 30 for video)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quitting program.")
            break
        elif key == ord("p"): # Pause/Play (useful for debugging, less common for continuous camera)
            cv2.waitKey(-1) # Wait indefinitely until any key is pressed
        # Add a key to adjust ROI dynamically (more advanced)

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Application terminated.")

if __name__ == "__main__":
    main_camera()