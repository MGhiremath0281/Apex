from ultralytics import YOLO
import cv2

# Load your custom trained model
model = YOLO("runs/detect/train/weights/best.pt")

mode = "video"
file_path = r'c:\Users\adity\Desktop\OBJDETECT\data\firevid1.mp4'

if mode == "image":
    # ... (same as before, consider adding tracking if processing sequences of images)
    results = model.predict(source=file_path, conf=0.5, iou=0.6, imgsz=640, half=True, device='0', save=False, verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow("Fire Detection", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if mode == "video":
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {file_path}. Please check the path and file integrity.")
    else:
        # Optional: Get video properties for potential output video saving
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
        # out = cv2.VideoWriter('output_fire_detection.mp4', fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
        #                       (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        print("Starting video processing. Press 'q' to quit.")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or failed to read frame.")
                break

            frame_count += 1

            # Use model.track() for better video stability and object persistence
            # You might need to install 'bytetrack': pip install lap scipy
            results = model.track(source=frame,
                                  conf=0.4,       # Adjusted confidence for fire (tune this carefully)
                                  iou=0.5,        # IoU for NMS (tune this carefully)
                                  imgsz=640,      # Input image size (should match training or be optimal)
                                  half=True,      # Use half-precision inference (GPU only, faster)
                                  device='0',     # Specify GPU device (e.g., '0' for first GPU, or 'cpu')
                                  persist=True,   # Essential for tracking objects across frames
                                  tracker="bytetrack.yaml", # Choose a tracker: "bytetrack.yaml" or "botsort.yaml"
                                  verbose=False)  # Suppress prediction logs to keep console clean

            # Get the annotated frame (includes tracking IDs if enabled)
            annotated_frame = results[0].plot()

            # Display the frame
            cv2.imshow("Real-time Fire Detection (Press 'q' to quit)", annotated_frame)

            # Optional: Save the processed video frame
            # if 'out' in locals() and out.isOpened():
            #     out.write(annotated_frame)

            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting video playback.")
                break

        cap.release()
        # if 'out' in locals() and out.isOpened():
        #     out.release()
        cv2.destroyAllWindows()
        print("Video processing finished.")