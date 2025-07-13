from ultralytics import YOLO
import cv2
import time

def main():
    # Load YOLO model
    model = YOLO('yolov8x.pt')

    # Path to your video file (or 0 for webcam)
    video_path = r'C:\Users\adity\Desktop\OBJDETECT\data\landing_notclearr.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # For FPS calculation
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        # Run inference
        results = model.predict(frame, verbose=False)

        # Use YOLO's built-in plot function (draws boxes & labels)
        annotated_frame = results[0].plot()

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Get number of detections
        num_objects = len(results[0].boxes)

        # Add custom overlay text
        cv2.putText(annotated_frame, "Apex Object Detection", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Objects: {num_objects}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        # Show frame
        cv2.imshow("Cool YOLO Video", annotated_frame)

        # Press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
