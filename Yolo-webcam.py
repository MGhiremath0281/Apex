from ultralytics import YOLO
import cv2
import time
from collections import Counter, deque
import os

# --- Configuration Constants for Accuracy and Performance ---
# Set a higher confidence threshold to filter out less certain detections.
# Range: 0.0 to 1.0. Default is usually 0.25. 0.5 is a common good starting point.
# Increasing it further (e.g., 0.6, 0.7) will make detections more "accurate" but might miss faint objects.
CONFIDENCE_THRESHOLD = 0.55 # Increased slightly from 0.5 for potentially stricter detections

# Set NMS IoU threshold. Lower value makes NMS more aggressive, reducing duplicate boxes.
# Range: 0.0 to 1.0. Default is usually 0.7. Lowering it (e.g., 0.4, 0.5) can clean up overlapping boxes.
NMS_IOU_THRESHOLD = 0.5 # Lowered to encourage more aggressive merging of overlapping boxes

# Input image size for the model. Larger size can increase accuracy for small objects, but decreases FPS.
# Default is 640. Common alternatives: 320, 480 (faster), 960, 1280 (slower, more accurate).
# Adjust this based on your performance needs and object sizes.
IMG_SIZE = 960 # Increased for potentially higher accuracy, especially for small objects. Be aware of FPS impact!

FPS_SMOOTHING_WINDOW = 5          # Number of frames for FPS moving average
OBJECT_PERSISTENCE_FRAMES = 5     # How many frames an object is "remembered" in logs after last detection

# Function to clear the terminal screen
def clear_terminal():
    if os.name == 'nt': # For Windows
        _ = os.system('cls')
    else: # For macOS and Linux
        _ = os.system('clear')

def draw_minimal_hud(frame, fps, current_datetime):
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (10 + 1, 30 + 1), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, fps_text, (10, 30), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    
    time_text = current_datetime
    text_size = cv2.getTextSize(time_text, font, font_scale, thickness)[0]
    text_x = w - text_size[0] - 10
    
    cv2.putText(frame, time_text, (text_x + 1, 30 + 1), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, time_text, (text_x, 30), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame

def main():
    print("--- Starting APEX LIVE DETECTION SYSTEM ---")
    
    try:
        print(f"Attempting to load YOLO model from: C:\\Users\\adity\\Desktop\\OBJDETECT\\yolov8x.pt")
        model = YOLO(r'C:\Users\adity\Desktop\OBJDETECT\yolov8x.pt')
        print("YOLO model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Please ensure 'yolov8x.pt' exists at the specified path and Ultralytics is installed.")
        return

    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    while not cap.isOpened():
        print(f"Error: Could not open webcam at index {camera_index}. Please check connection or if it's in use. Retrying in 2 seconds...")
        time.sleep(2) # Wait a bit before retrying
        cap = cv2.VideoCapture(camera_index) # Re-attempt to open

    print(f"Webcam opened successfully at index {camera_index}.")

    frame_times = deque(maxlen=FPS_SMOOTHING_WINDOW)
    
    persistent_objects = {} 
    current_frame_idx = 0

    clear_terminal()
    print("--- APEX LIVE DETECTION SYSTEM (Terminal Log) ---")
    print("Status: Running...")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"NMS IoU Threshold: {NMS_IOU_THRESHOLD}")
    print(f"Input Image Size (imgsz): {IMG_SIZE}")
    print("-------------------------------------------------")
    print("Press 'q' in the webcam window to quit.")
    
    time.sleep(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nFailed to grab frame. Check webcam connection or if it's blocked. Exiting...")
            break
        
        current_frame_idx += 1

        try:
            # --- Key changes for accuracy and prediction here ---
            results = model.predict(
                frame, 
                device=0,          # Use GPU. Change to 'cpu' if no compatible GPU.
                verbose=False,
                conf=CONFIDENCE_THRESHOLD, # Filter detections by confidence
                iou=NMS_IOU_THRESHOLD,     # Adjust NMS for overlapping boxes
                imgsz=IMG_SIZE,            # Set input image size for prediction
                half=True                  # Use half-precision inference (for compatible GPUs)
            )
            result = results[0] 
        except Exception as e:
            print(f"Error during model prediction: {e}")
            print("This might be a GPU/CUDA issue or model problem. Exiting.")
            break

        annotated_frame = result.plot()

        curr_time = time.time()
        frame_times.append(curr_time)
        if len(frame_times) > 1:
            avg_frame_time = (frame_times[-1] - frame_times[0]) / (len(frame_times) - 1)
            fps = 1 / avg_frame_time if avg_frame_time > 0 else 0
        else:
            fps = 0 
        
        current_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        final_frame = draw_minimal_hud(annotated_frame, fps, current_datetime)

        current_frame_objects = Counter()
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = result.names[class_id]
            current_frame_objects[class_name] += 1
            persistent_objects[class_name] = current_frame_idx

        objects_to_log = Counter()
        keys_to_remove = []
        for cls_name, last_frame_idx in persistent_objects.items():
            if (current_frame_idx - last_frame_idx) <= OBJECT_PERSISTENCE_FRAMES:
                objects_to_log[cls_name] = current_frame_objects.get(cls_name, 0) 
            else:
                keys_to_remove.append(cls_name)
        
        for key in keys_to_remove:
            del persistent_objects[key]

        for cls_name, count in current_frame_objects.items():
             objects_to_log[cls_name] = count

        clear_terminal()

        print("--- APEX LIVE DETECTION SYSTEM (Terminal Log) ---")
        print(f"  Last Update: {current_datetime}")
        print(f"  Smoothed FPS: {fps:.1f}")
        print(f"  Total Active Objects: {sum(objects_to_log.values())}")
        print(f"  --- Model Parameters ---")
        print(f"  Confidence: {CONFIDENCE_THRESHOLD}")
        print(f"  NMS IoU: {NMS_IOU_THRESHOLD}")
        print(f"  Input Size: {IMG_SIZE}")
        print("-------------------------------------------------")
        
        if len(objects_to_log) > 0:
            print("  PERSISTENT OBJECT COUNTS:")
            for cls_name, count in sorted(objects_to_log.items()):
                print(f"    - {cls_name.upper()}: {count}")
        else:
            print("  No persistent objects detected.")
        print("\n-------------------------------------------------")
        print("Press 'q' in the webcam window to quit.")

        cv2.imshow("APEX LIVE DETECTION - Webcam Feed", final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n'q' pressed. Exiting application. Cleaning up...")
            break

    clear_terminal()
    print("APEX LIVE DETECTION SYSTEM - Session Ended.")
    print("Resources released. Goodbye!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()