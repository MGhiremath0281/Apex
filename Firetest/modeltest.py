from ultralytics import YOLO
import cv2
import time
import torch

# Load your trained model
model = YOLO("yolofirenew.pt").to('cuda' if torch.cuda.is_available() else 'cpu')

# MODE and FILE path
mode = "video"
file_path = r'c:\Users\adity\Desktop\OBJDETECT\data\firevid1.mp4'

# --- Optimization Parameters ---
TARGET_IMG_SIZE = 320     # Smaller sizes = faster inference, less accuracy
SKIP_FRAMES = 2           # Skip every Nth frame to speed up

if mode == "image":
    results = model.predict(
        source=file_path,
        conf=0.5,
        iou=0.6,
        imgsz=TARGET_IMG_SIZE,
        device='cpu',
        save=False,
        verbose=False
    )
    annotated_frame = results[0].plot()
    cv2.imshow("Fire Detection", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif mode == "video":
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {file_path}")
        exit()

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Original video FPS: {original_fps:.2f}")
    print(f"Processing every {SKIP_FRAMES} frames. Press 'q' to quit.")

    frame_count = 0
    inference_times = []
    display_start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        frame_count += 1

        # Skip frames
        if frame_count % SKIP_FRAMES != 0:
            continue

        # Run YOLOv8 prediction
        start_time = time.time()
        results = model.predict(
            source=frame,
            conf=0.4,
            iou=0.5,
            imgsz=TARGET_IMG_SIZE,
            device='cpu',
            verbose=False
        )
        end_time = time.time()

        inference_time = end_time - start_time
        inference_times.append(inference_time)

        # Draw predictions
        annotated_frame = results[0].plot()

        # Display FPS and timing info
        display_time = time.time() - display_start_time
        approx_displayed_frames = frame_count / SKIP_FRAMES
        if display_time > 0:
            fps = approx_displayed_frames / display_time
            cv2.putText(annotated_frame, f"Displayed FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if inference_times:
            avg_inf_ms = sum(inference_times) / len(inference_times) * 1000
            cv2.putText(annotated_frame, f"Inference: {avg_inf_ms:.2f} ms", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Real-time Fire Detection (press 'q' to quit)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exited by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nVideo processing complete.")
    print(f"Total frames read: {frame_count}")
    print(f"Frames processed: {len(inference_times)}")
    if inference_times:
        avg = sum(inference_times) / len(inference_times)
        print(f"Average inference time: {avg:.4f} seconds ({avg*1000:.2f} ms)")
