import cv2
import time
import os
import pyttsx3
from ultralytics import YOLO

# --- Configuration ---
WARNING_MESSAGE = "Unauthorized object detected! Please take action immediately."

# YOLO Model Configuration
YOLO_MODEL_PATH = "yolov8x.pt"  # Using official YOLOv8x model
CONFIDENCE_THRESHOLD = 0.5

# --- Define your classes directly here (COCO classes by default) ---
class_names = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", 
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", 
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", 
    "backpack", "umbrella", "handbag", "tie", "suitcase", 
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", 
    "baseball glove", "skateboard", "surfboard", "tennis racket", 
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", 
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", 
    "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", 
    "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", 
    "sink", "refrigerator", "book", "clock", "vase", "scissors", 
    "teddy bear", "hair drier", "toothbrush"
]

# Define what objects trigger a warning
FORBIDDEN_OBJECTS = [
    "person",       # Example: warn if a person is detected
    "car",          # Example: warn if a car is detected
    "truck"         # Example: warn if a truck is detected
]

# --- Global variable to prevent repeated warnings ---
last_warning_time = 0
WARNING_COOLDOWN_SECONDS = 10

# --- Load YOLO Model ---
try:
    model = YOLO(YOLO_MODEL_PATH)
    print(f"YOLO model loaded from: {YOLO_MODEL_PATH}")
    print(f"Loaded {len(class_names)} classes (defined in code).")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# --- Text-to-Speech Function ---
def speak_with_pyttsx3(text):
    global last_warning_time
    if time.time() - last_warning_time < WARNING_COOLDOWN_SECONDS:
        return

    try:
        print(f"pyttsx3: Speaking: '{text}'")
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        last_warning_time = time.time()
    except Exception as e:
        print(f"Error with pyttsx3: {e}")

# --- YOLO Integration Logic ---
def process_frame_for_forbidden_objects(frame, model, class_names, forbidden_objects, confidence_threshold):
    results = model(frame, verbose=False)[0]

    detected_forbidden_objects = []
    for r in results.boxes:
        conf = float(r.conf)
        cls = int(r.cls)
        
        if conf >= confidence_threshold:
            detected_class_name = class_names[cls]
            if detected_class_name in forbidden_objects:
                detected_forbidden_objects.append(detected_class_name)
    
    return len(detected_forbidden_objects) > 0, detected_forbidden_objects

def main():
    print("Voice Warning Integration with YOLO Object Detection")
    print("--------------------------------------------------")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream or file.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        is_breach, detected_objects = process_frame_for_forbidden_objects(
            frame.copy(),
            model,
            class_names,
            FORBIDDEN_OBJECTS,
            CONFIDENCE_THRESHOLD
        )

        results_visual = model(frame, verbose=False)[0]
        annotated_frame = results_visual.plot()

        if is_breach:
            print(f"!!! Forbidden object(s) detected: {', '.join(detected_objects)} !!!")
            speak_with_pyttsx3(WARNING_MESSAGE)

        cv2.imshow("YOLO Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
