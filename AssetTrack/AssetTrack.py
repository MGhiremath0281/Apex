import cv2
import numpy as np
import json
import time
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict, deque
import math
import queue


class AssetTracker:
    def __init__(self, model_path='yolov8x.pt', config_file='asset_config.json'):
        self.model = YOLO(model_path)
        self.load_config(config_file)

        self.tracked_objects = {}
        self.next_id = 1
        self.trajectory_history = defaultdict(lambda: deque(maxlen=50))
        self.max_disappeared = 30
        self.max_distance = 100
        self.confidence_threshold = 0.5

        self.alerts = queue.Queue()
        self.alert_cooldown = {}
        self.alert_duration = 5.0

        self.frame_count = 0
        self.fps = 30

        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 0)
        ]

    def load_config(self, config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.asset_definitions = config.get('assets', {})
            self.workstations = config.get('workstations', {})
            self.allowed_locations = config.get('allowed_locations', {})
        except FileNotFoundError:
            self.create_default_config(config_file)

    def create_default_config(self, config_file):
        default_config = {
            "assets": {
                "wrench": {"id": "TOOL_001", "type": "tool", "workstation": "station_1"},
                "hammer": {"id": "TOOL_002", "type": "tool", "workstation": "station_2"},
                "screwdriver": {"id": "TOOL_003", "type": "tool", "workstation": "station_1"},
                "drill": {"id": "TOOL_004", "type": "tool", "workstation": "station_3"},
                "pliers": {"id": "TOOL_005", "type": "tool", "workstation": "station_2"}
            },
            "workstations": {
                "station_1": {"area": [50, 50, 300, 200], "name": "Assembly Station"},
                "station_2": {"area": [350, 50, 600, 200], "name": "Repair Station"},
                "station_3": {"area": [650, 50, 900, 200], "name": "Quality Control"}
            },
            "allowed_locations": {
                "TOOL_001": ["station_1", "station_2"],
                "TOOL_002": ["station_2", "station_3"],
                "TOOL_003": ["station_1"],
                "TOOL_004": ["station_3"],
                "TOOL_005": ["station_2"]
            }
        }
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        self.asset_definitions = default_config['assets']
        self.workstations = default_config['workstations']
        self.allowed_locations = default_config['allowed_locations']

    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def get_workstation_from_position(self, x, y):
        for station_id, data in self.workstations.items():
            x1, y1, x2, y2 = data['area']
            if x1 <= x <= x2 and y1 <= y <= y2:
                return station_id
        return None

    def is_location_allowed(self, asset_id, current_station):
        if asset_id not in self.allowed_locations:
            return True
        return current_station in self.allowed_locations[asset_id]

    def associate_detections_to_objects(self, detections):
        if not self.tracked_objects:
            return [None] * len(detections)

        object_centers = [(obj_id, obj_data['center'])
                          for obj_id, obj_data in self.tracked_objects.items()
                          if obj_data['disappeared'] < self.max_disappeared]

        assignments = []
        used_objects = set()

        for det in detections:
            det_center = det['center']
            best_match = None
            min_distance = float('inf')

            for obj_id, obj_center in object_centers:
                if obj_id in used_objects:
                    continue
                distance = self.calculate_distance(det_center, obj_center)
                if distance < min_distance and distance < self.max_distance:
                    best_match = obj_id
                    min_distance = distance

            if best_match is not None:
                assignments.append(best_match)
                used_objects.add(best_match)
            else:
                assignments.append(None)

        return assignments

    def update_tracked_objects(self, detections, assignments):
        for obj_id in self.tracked_objects:
            self.tracked_objects[obj_id]['disappeared'] += 1

        for i, detection in enumerate(detections):
            assigned_id = assignments[i]

            if assigned_id is None or assigned_id not in self.tracked_objects:
                obj_id = self.next_id
                self.next_id += 1

                asset_id = None
                class_name = detection['class']
                if class_name in self.asset_definitions:
                    asset_id = self.asset_definitions[class_name]['id']

                self.tracked_objects[obj_id] = {
                    'asset_id': asset_id,
                    'class': class_name,
                    'center': detection['center'],
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'disappeared': 0,
                    'first_seen': self.frame_count,
                    'last_seen': self.frame_count
                }
            else:
                obj_id = assigned_id
                self.tracked_objects[obj_id].update({
                    'center': detection['center'],
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'disappeared': 0,
                    'last_seen': self.frame_count
                })

            self.trajectory_history[obj_id].append(detection['center'])

    def check_alerts(self):
        current_time = time.time()
        for obj_id, obj_data in self.tracked_objects.items():
            if obj_data['disappeared'] > 0:
                continue
            asset_id = obj_data['asset_id']
            if asset_id is None:
                continue
            center = obj_data['center']
            current_station = self.get_workstation_from_position(*center)
            if current_station and not self.is_location_allowed(asset_id, current_station):
                alert_key = f"misplaced_{asset_id}"
                if alert_key not in self.alert_cooldown or current_time - self.alert_cooldown[alert_key] > self.alert_duration:
                    self.alerts.put({
                        'type': 'misplaced',
                        'asset_id': asset_id,
                        'class': obj_data['class'],
                        'current_station': current_station,
                        'allowed_stations': self.allowed_locations.get(asset_id, []),
                        'timestamp': current_time,
                        'position': center
                    })
                    self.alert_cooldown[alert_key] = current_time

        for asset_name, asset_data in self.asset_definitions.items():
            asset_id = asset_data['id']
            found = False
            for obj_data in self.tracked_objects.values():
                if obj_data['asset_id'] == asset_id and obj_data['disappeared'] < self.max_disappeared:
                    found = True
                    break
            if not found:
                alert_key = f"missing_{asset_id}"
                if alert_key not in self.alert_cooldown or current_time - self.alert_cooldown[alert_key] > self.alert_duration:
                    self.alerts.put({
                        'type': 'missing',
                        'asset_id': asset_id,
                        'class': asset_name,
                        'expected_station': asset_data.get('workstation', 'unknown'),
                        'timestamp': current_time
                    })
                    self.alert_cooldown[alert_key] = current_time

    def draw_trajectory(self, frame, obj_id, color):
        trajectory = list(self.trajectory_history[obj_id])
        for i, point in enumerate(trajectory):
            alpha = 0.3 + (i / len(trajectory)) * 0.7
            dot_color = tuple(int(c * alpha) for c in color)
            cv2.circle(frame, (int(point[0]), int(point[1])), 3, dot_color, -1)

        for i in range(len(trajectory) - 1):
            start = tuple(map(int, trajectory[i]))
            end = tuple(map(int, trajectory[i + 1]))
            if self.calculate_distance(start, end) > 10:
                cv2.arrowedLine(frame, start, end, color, 2, tipLength=0.3)

    def draw_workstations(self, frame):
        for station_id, data in self.workstations.items():
            x1, y1, x2, y2 = data['area']
            name = data['name']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 2)
            cv2.putText(frame, name, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def draw_alerts(self, frame):
        alert_y = 50
        temp_alerts = []
        while not self.alerts.empty():
            try:
                temp_alerts.append(self.alerts.get_nowait())
            except queue.Empty:
                break

        for alert in temp_alerts:
            if alert['type'] == 'misplaced':
                text = f"ALERT: {alert['class']} ({alert['asset_id']}) misplaced in {alert['current_station']}"
                color = (0, 0, 255)
            elif alert['type'] == 'missing':
                text = f"ALERT: {alert['class']} ({alert['asset_id']}) missing from {alert['expected_station']}"
                color = (0, 165, 255)
            else:
                continue
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (10, alert_y - 25), (size[0] + 20, alert_y + 10), color, -1)
            cv2.putText(frame, text, (15, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            alert_y += 40

    def process_frame(self, frame):
        self.frame_count += 1
        results = self.model(frame, conf=self.confidence_threshold)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center': center,
                    'confidence': confidence,
                    'class': class_name,
                    'class_id': class_id
                })

        assignments = self.associate_detections_to_objects(detections)
        self.update_tracked_objects(detections, assignments)
        self.check_alerts()

        to_remove = [obj_id for obj_id, obj in self.tracked_objects.items() if obj['disappeared'] > self.max_disappeared]
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
            self.trajectory_history.pop(obj_id, None)

        return self.draw_frame(frame)

    def draw_frame(self, frame):
        self.draw_workstations(frame)
        for obj_id, obj_data in self.tracked_objects.items():
            if obj_data['disappeared'] > 0:
                continue
            color = self.colors[obj_id % len(self.colors)]
            bbox = obj_data['bbox']
            center = obj_data['center']
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.circle(frame, (int(center[0]), int(center[1])), 5, color, -1)
            self.draw_trajectory(frame, obj_id, color)
            label = f"{obj_data['class']}"
            if obj_data['asset_id']:
                label += f" ({obj_data['asset_id']})"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        self.draw_alerts(frame)
        cv2.putText(frame, f"Frame: {self.frame_count} | Objects: {len(self.tracked_objects)}",
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame


def main():
    tracker = AssetTracker()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Asset Tracking System Started")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        processed_frame = tracker.process_frame(frame)
        cv2.imshow("Asset Tracking System", processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, processed_frame)
            print(f"Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("Asset Tracking System Stopped")


if __name__ == "__main__":
    main()
