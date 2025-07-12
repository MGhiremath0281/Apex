import cv2

print("Checking available camera indices...")

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Camera index {i} is available.")
        cap.release()
    else:
        print(f"❌ Camera index {i} is not available.")
