from ultralytics import YOLO
import cv2
from pathlib import Path

def main():
    # Load pre-trained YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    # Your image path
    image_path = r'C:\Users\adity\Desktop\OBJDETECT\data\test1.jpg'

    # Run prediction
    results = model(image_path, save=True)

    # Print readable class names and confidence scores
    print("Detected objects:")
    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = results[0].names[class_id]
        confidence = float(box.conf)
        print(f" - {class_name} ({confidence:.2f})")

    print(f"\nTotal objects detected: {len(results[0].boxes)}")

    # Corrected saved_path construction
    saved_path = Path(results[0].save_dir) / Path(results[0].path).name

    # Show the image using OpenCV
    img = cv2.imread(str(saved_path))
    cv2.imshow("Detection Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
