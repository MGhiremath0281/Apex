from ultralytics import YOLO
import cv2
import time

# Load your custom trained model
# If your 'best.pt' is a large model (e.g., yolov8l, yolov8x),
# for CPU speed, consider training/using a smaller model like yolov8n.pt or yolov8s.pt
# model = YOLO("yolov8n.pt") # Uncomment to try a pre-trained nano model for max speed
model = YOLO("runs/detect/train/weights/best.pt")

mode = "video"
file_path = r'c:\Users\adity\Desktop\OBJDETECT\data\firevid1.mp4'

# --- Optimization Parameters for Speed ---
# Adjust TARGET_IMG_SIZE: Smaller images are processed much faster on CPU.
# Common sizes: 320, 480, 640. Start low for speed, increase if accuracy suffers.
TARGET_IMG_SIZE = 320 # Try 320 or 480 for significant speed up on CPU

# Adjust SKIP_FRAMES: This directly speeds up the video playback by skipping analysis on frames.
# 1 = process every frame (no skipping)
# 2 = process every 2nd frame (skips 1, processes 1) - effectively doubles playback speed
# 3 = process every 3rd frame - triples playback speed
# ... and so on.
# Be aware: Skipping frames means you might miss very brief events.
SKIP_FRAMES = 2 # Start with 2, then try 3 or 4 if still too slow.

if mode == "image":
    # For image prediction, also specify device='cpu'
    results = model.predict(source=file_path, conf=0.5, iou=0.6, imgsz=TARGET_IMG_SIZE, device='cpu', save=False, verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow("Fire Detection", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if mode == "video":
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {file_path}. Please check the path and file integrity.")
    else:
        # Get original video FPS for reference
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Original video FPS: {original_fps:.2f}")
        print(f"Starting video processing. Press 'q' to quit. Processing every {SKIP_FRAMES} frames.")

        frame_count = 0
        inference_times = [] # To track actual inference duration
        display_start_time = time.time() # To calculate overall displayed FPS

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or failed to read frame.")
                break

            frame_count += 1

            # --- Frame Skipping Logic ---
            # If we're skipping frames, we only process specific ones
            if frame_count % SKIP_FRAMES != 0:
                # If you want to still display the skipped frames (without detection)
                # you would move cv2.imshow() here and display 'frame' directly.
                # For speeding up detection, we just skip processing entirely.
                continue

            # --- Perform Inference ---
            start_inference_time = time.time()
            results = model.track(source=frame,
                                  conf=0.4,
                                  iou=0.5,
                                  imgsz=TARGET_IMG_SIZE,
                                  device='cpu',
                                  persist=True,
                                  tracker="bytetrack.yaml",
                                  verbose=False)
            end_inference_time = time.time()
            inference_times.append(end_inference_time - start_inference_time)

            annotated_frame = results[0].plot()

            # --- Display FPS and Inference Time ---
            current_display_time = time.time()
            total_elapsed_display_time = current_display_time - display_start_time
            
            # Calculate overall displayed FPS (how fast the video appears to play)
            # This is the number of frames displayed divided by elapsed time
            displayed_frames = frame_count / SKIP_FRAMES # Approximate
            if total_elapsed_display_time > 0:
                displayed_fps = displayed_frames / total_elapsed_display_time
                cv2.putText(annotated_frame, f"Displayed FPS: {displayed_fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Calculate average inference time per processed frame
            if inference_times:
                avg_inference_time = sum(inference_times) / len(inference_times)
                cv2.putText(annotated_frame, f"Inference (ms): {avg_inference_time * 1000:.2f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Real-time Fire Detection (Press 'q' to quit)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting video playback.")
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Video processing finished.")
        if inference_times:
            print(f"Average inference time per processed frame: {sum(inference_times) / len(inference_times):.4f} seconds")
        print(f"Total frames read: {frame_count}")
        print(f"Total frames processed for detection: {len(inference_times)}")