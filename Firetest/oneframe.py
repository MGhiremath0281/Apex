import cv2
from ultralytics import YOLO
import os # For checking file existence

# --- Model Loading ---
# Load the general COCO model
person_vehicle_model = YOLO('yolov8x.pt')

# Load your custom fire/smoke model
# IMPORTANT: Replace 'path_to_your_fire_model.pt' with the actual path to your trained fire model
fire_model = YOLO('runs/detect/train/weights/best.pt')

# --- Configuration ---
# Set the input source:
# For an image: "test_image.jpg"
# For a video file: "test_video.mp4"
# For webcam: 0 (or 1, 2, etc., depending on your webcam ID)
input_source = r'c:\Users\adity\Desktop\OBJDETECT\data\firevid1.mp4'

# Output folder for saving results (optional)
output_folder = "output_detections"
os.makedirs(output_folder, exist_ok=True)

# Function to process a single frame (image)
def process_frame(frame, frame_idx=None, is_video=False):
    """
    Processes a single image frame for object detection and draws results.

    Args:
        frame (np.array): The input image frame (BGR format).
        frame_idx (int, optional): The index of the frame if processing a video. Defaults to None.
        is_video (bool): True if processing a video, False if processing a single image.
    Returns:
        np.array: The frame with drawn detections.
    """
    # Clone the frame to draw on it, keeping the original intact if needed elsewhere
    display_frame = frame.copy()

    # Run both models for inference
    # Using verbose=False to suppress print statements during inference
    results_general = person_vehicle_model(display_frame, verbose=False)
    results_fire = fire_model(display_frame, verbose=False)

    # Get the Boxes objects from the results
    boxes_general = results_general[0].boxes
    boxes_fire = results_fire[0].boxes

    # --- Draw detections from the general model (e.g., green boxes) ---
    for box in boxes_general:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get integer coordinates
        conf = box.conf[0].item()               # Confidence score
        cls = int(box.cls[0].item())            # Class ID
        label = person_vehicle_model.names[cls] # Get class name

        # Filter out fire/smoke classes if they are also detected by the general model
        # You might need to adjust class names based on your fire_model.names
        if label not in ['fire', 'smoke', 'flame', 'blaze']: # Added more fire-related terms
            color = (0, 255, 0)  # Green color for general objects
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # --- Draw detections from the fire model (e.g., red boxes) ---
    for box in boxes_fire:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get integer coordinates
        conf = box.conf[0].item()               # Confidence score
        cls = int(box.cls[0].item())            # Class ID
        label = fire_model.names[cls]           # Get class name from your fire model's names

        color = (0, 0, 255)  # Red color for fire/smoke
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return display_frame

# --- Main Execution Logic ---
if isinstance(input_source, str) and (input_source.endswith('.jpg') or input_source.endswith('.png')):
    # --- Process Image ---
    print(f"Processing image: {input_source}")
    img = cv2.imread(input_source)

    if img is None:
        print(f"Error: Could not load image from {input_source}. Please check the path.")
    else:
        processed_img = process_frame(img)
        cv2.imshow("Combined Detections (Image)", processed_img)
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows()
        # Save the processed image
        cv2.imwrite(os.path.join(output_folder, os.path.basename(input_source).replace('.', '_combined.')), processed_img)
        print(f"Processed image saved to {os.path.join(output_folder, os.path.basename(input_source).replace('.', '_combined.'))}")

elif isinstance(input_source, (str, int)): # Could be video file path or webcam ID
    # --- Process Video/Webcam ---
    cap = cv2.VideoCapture(input_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {input_source}. Check file path or webcam ID.")
    else:
        print(f"Processing video/webcam: {input_source}")
        # Optional: Setup video writer if saving output video
        out = None
        if isinstance(input_source, str) and (input_source.endswith('.mp4') or input_source.endswith('.avi')):
            # Get video properties for output
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_video_path = os.path.join(output_folder, os.path.basename(input_source).replace('.', '_combined.'))
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            print(f"Saving output video to {output_video_path}")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or error reading frame.")
                break

            frame_count += 1
            processed_frame = process_frame(frame, frame_idx=frame_count, is_video=True)

            cv2.imshow("Combined Detections (Video)", processed_frame)

            if out is not None:
                out.write(processed_frame)

            # Press 'q' to quit the video stream
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        print("Video processing finished.")

else:
    print("Invalid input_source. Please specify an image path, video path, or webcam ID (0, 1, ...).")