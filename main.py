from ultralytics import YOLO
import cv2
import numpy as np
import time
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- CONFIG ---
ALERT_TIME = 60  # seconds a bag must be left alone to trigger alert
DIST_THRESHOLD = 100  # pixels (distance to consider "close")
# -----------------

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # or yolov8s.pt

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Open webcam
cap = cv2.VideoCapture(0)

# Dictionary to keep track of bag ownership
bag_owners = {}  # bag_id: (owner_id, last_seen_time)
bag_timers = {}  # bag_id: start_time

# Define COCO classes for reference
COCO_CLASSES = model.names

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if COCO_CLASSES[cls] in ['person', 'backpack', 'handbag', 'suitcase']:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, COCO_CLASSES[cls]))

    tracks = tracker.update_tracks(detections, frame=frame)

    persons = {}
    bags = {}

    # Separate tracked objects
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cls = track.get_det_class()

        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if cls == "person":
            persons[track_id] = center
        elif cls in ["backpack", "handbag", "suitcase"]:
            bags[track_id] = center

        # Draw
        color = (0, 255, 0) if cls == "person" else (255, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{cls} {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Check bag ownership or start unattended timer
    current_time = time.time()

    for bag_id, bag_center in bags.items():
        # Find closest person
        min_dist = float('inf')
        closest_person_id = None
        for person_id, person_center in persons.items():
            dist = np.linalg.norm(np.array(bag_center) - np.array(person_center))
            if dist < min_dist:
                min_dist = dist
                closest_person_id = person_id

        if bag_id not in bag_owners:
            if min_dist < DIST_THRESHOLD:
                # Assign this person as the bag owner
                bag_owners[bag_id] = (closest_person_id, current_time)
                if bag_id in bag_timers:
                    bag_timers.pop(bag_id)
        else:
            owner_id, _ = bag_owners[bag_id]
            # Check if the owner is still close
            if owner_id in persons:
                dist = np.linalg.norm(np.array(bag_center) - np.array(persons[owner_id]))
                if dist < DIST_THRESHOLD:
                    bag_owners[bag_id] = (owner_id, current_time)
                    if bag_id in bag_timers:
                        bag_timers.pop(bag_id)
                else:
                    # Owner far away
                    if bag_id not in bag_timers:
                        bag_timers[bag_id] = current_time
            else:
                # Owner not in frame
                if bag_id not in bag_timers:
                    bag_timers[bag_id] = current_time

    # Draw alerts
    for bag_id, start_time in bag_timers.items():
        elapsed = current_time - start_time
        if elapsed >= ALERT_TIME:
            # Highlight bag in red
            if bag_id in bags:
                cx, cy = bags[bag_id]
                cv2.putText(frame, "ALERT! Unattended bag!", (cx - 50, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                cv2.circle(frame, (cx, cy), 40, (0, 0, 255), 4)

    cv2.imshow("Unattended Bag Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
