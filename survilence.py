import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque, defaultdict
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Detection:
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    center: Tuple[int, int]
    assigned_person_id: Optional[int] = None

class TrajectoryTracker:
    def __init__(self, max_trail_length=30):
        self.trails = defaultdict(lambda: deque(maxlen=max_trail_length))
        self.colors = {}

    def update(self, track_id: int, center: Tuple[int, int]):
        self.trails[track_id].append(center)
        if track_id not in self.colors:
            np.random.seed(track_id)
            self.colors[track_id] = (
                int(np.random.randint(0, 255)),
                int(np.random.randint(0, 255)),
                int(np.random.randint(0, 255))
            )

    def draw_trails(self, frame: np.ndarray):
        for track_id, trail in self.trails.items():
            if len(trail) > 1:
                color = self.colors[track_id]
                points = np.array(trail, dtype=np.int32)
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    thickness = max(1, int(3 * alpha))
                    cv2.line(frame, tuple(points[i-1]), tuple(points[i]), color, thickness)
                if len(points) >= 2:
                    self.draw_arrow(frame, points[-2], points[-1], color)

    def draw_arrow(self, frame: np.ndarray, start: np.ndarray, end: np.ndarray, color: tuple):
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        arrow_length = 10
        arrow_angle = math.pi / 6
        x1 = int(end[0] - arrow_length * math.cos(angle - arrow_angle))
        y1 = int(end[1] - arrow_length * math.sin(angle - arrow_angle))
        x2 = int(end[0] - arrow_length * math.cos(angle + arrow_angle))
        y2 = int(end[1] - arrow_length * math.sin(angle + arrow_angle))
        cv2.arrowedLine(frame, tuple(start), tuple(end), color, 2)
        cv2.line(frame, tuple(end), (x1, y1), color, 2)
        cv2.line(frame, tuple(end), (x2, y2), color, 2)

class PersonBagTracker:
    def __init__(self, model_path='yolov8x.pt'):
        self.model = YOLO(model_path)
        self.trajectory_tracker = TrajectoryTracker()
        self.person_class_id = 0
        self.bag_class_ids = {24: 'handbag', 26: 'backpack', 28: 'suitcase'}
        self.active_persons: Set[int] = set()
        self.active_bags: Set[int] = set()
        self.frame_count = 0
        self.bag_id_mapping: Dict[int, int] = {}
        self.bag_last_person_id: Dict[int, int] = {}
        self.bag_association_frames: Dict[int, int] = defaultdict(int)
        self.unattended_bags: Dict[int, int] = {}
        self.alert_threshold = 30  # frames before alert
        self.alerts_logged: Set[int] = set()
        self.max_association_distance = 150
        self.holding_margin = 30  # pixels for "holding" (bag inside person bbox)
        self.sticky_frames = 15   # Number of frames to keep association after separation
        logger.info("PersonBagTracker initialized with YOLOv8x model")

    @staticmethod
    def calculate_distance(center1: Tuple[int, int], center2: Tuple[int, int]) -> float:
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    @staticmethod
    def is_bag_held_by_person(person_bbox, bag_bbox, margin=30):
        px1, py1, px2, py2 = person_bbox
        bx1, by1, bx2, by2 = bag_bbox
        # Bag is "held" if its center is inside the person's bbox (with margin)
        bag_center = ((bx1 + bx2) // 2, (by1 + by2) // 2)
        return (px1 - margin <= bag_center[0] <= px2 + margin and
                py1 - margin <= bag_center[1] <= py2 + margin)

    def detect_and_track(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.track(frame, persist=True, verbose=False)
        detections = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            current_active_persons = set()
            current_active_bags = set()
            for box, track_id, class_id, conf in zip(boxes, ids, classes, confidences):
                x1, y1, x2, y2 = map(int, box)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                class_name = self.model.names[class_id]
                if class_id == self.person_class_id:
                    detection = Detection(
                        track_id=track_id,
                        class_id=class_id,
                        class_name=class_name,
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        center=center,
                        assigned_person_id=track_id
                    )
                    detections.append(detection)
                    self.trajectory_tracker.update(track_id, center)
                    current_active_persons.add(track_id)
                elif class_id in self.bag_class_ids:
                    detection = Detection(
                        track_id=track_id,
                        class_id=class_id,
                        class_name=class_name,
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        center=center,
                        assigned_person_id=None
                    )
                    detections.append(detection)
                    self.trajectory_tracker.update(track_id, center)
                    current_active_bags.add(track_id)
            self.active_persons = current_active_persons
            self.active_bags = current_active_bags
        return detections

    def update_associations(self, detections: List[Detection]):
        persons_in_frame = {d.track_id: d for d in detections if d.class_id == self.person_class_id}
        bags_in_frame = {d.track_id: d for d in detections if d.class_id in self.bag_class_ids}

        # For each bag, find the closest person (if within threshold) and check if "held"
        for bag_id, bag_det in bags_in_frame.items():
            best_person_id = None
            best_dist = float('inf')
            is_held = False
            for person_id, person_det in persons_in_frame.items():
                dist = self.calculate_distance(person_det.center, bag_det.center)
                held = self.is_bag_held_by_person(person_det.bbox, bag_det.bbox, self.holding_margin)
                if held:
                    is_held = True
                    best_person_id = person_id
                    best_dist = dist
                    break  # Prefer "held" association
                elif dist < self.max_association_distance and dist < best_dist:
                    best_person_id = person_id
                    best_dist = dist

            # Association logic
            if best_person_id is not None:
                # If "held", make association very sticky
                if is_held:
                    self.bag_association_frames[bag_id] = self.sticky_frames * 2
                else:
                    self.bag_association_frames[bag_id] = self.sticky_frames
                self.bag_id_mapping[bag_id] = best_person_id
                self.bag_last_person_id[bag_id] = best_person_id
            else:
                # No person nearby, decrease stickiness
                self.bag_association_frames[bag_id] -= 1
                if self.bag_association_frames[bag_id] <= 0:
                    self.bag_id_mapping[bag_id] = bag_id  # Unassigned

        # Remove associations for bags no longer in frame
        for bag_id in list(self.bag_id_mapping.keys()):
            if bag_id not in self.active_bags:
                del self.bag_id_mapping[bag_id]
                if bag_id in self.bag_association_frames:
                    del self.bag_association_frames[bag_id]
                if bag_id in self.bag_last_person_id:
                    del self.bag_last_person_id[bag_id]

        # Assign IDs to detections for drawing/alerting
        for det in detections:
            if det.class_id in self.bag_class_ids:
                det.assigned_person_id = self.bag_id_mapping.get(det.track_id, det.track_id)

    def check_unattended_bags(self):
        for bag_id in self.active_bags:
            associated_person_id = self.bag_id_mapping.get(bag_id, bag_id)
            # If bag is not mapped to a person (or mapped to itself)
            if associated_person_id == bag_id:
                # Standard unattended bag logic
                if bag_id not in self.unattended_bags:
                    self.unattended_bags[bag_id] = self.frame_count
                    logger.info(f"Bag {bag_id} detected as potentially unattended (no active association).")
                frames_unattended = self.frame_count - self.unattended_bags[bag_id]
                if frames_unattended >= self.alert_threshold and bag_id not in self.alerts_logged:
                    owner_id_for_alert = self.bag_last_person_id.get(bag_id, "UNKNOWN")
                    self.trigger_alert(bag_id, owner_id_for_alert)
                    self.alerts_logged.add(bag_id)
            else:
                # Bag is associated with a person, but check if the person is still present
                if associated_person_id not in self.active_persons:
                    # The owner has left the scene!
                    if bag_id not in self.unattended_bags:
                        self.unattended_bags[bag_id] = self.frame_count
                        logger.info(f"Bag {bag_id} is now unattended because owner {associated_person_id} left the scene.")
                    frames_unattended = self.frame_count - self.unattended_bags[bag_id]
                    if frames_unattended >= self.alert_threshold and bag_id not in self.alerts_logged:
                        self.trigger_alert(bag_id, associated_person_id)
                        self.alerts_logged.add(bag_id)
                else:
                    # Bag is attended
                    if bag_id in self.unattended_bags:
                        del self.unattended_bags[bag_id]
                        logger.info(f"Bag {bag_id} is no longer unattended.")
                    if bag_id in self.alerts_logged:
                        self.alerts_logged.remove(bag_id)
        # Clean up for bags that left the frame
        for bag_id in list(self.unattended_bags.keys()):
            if bag_id not in self.active_bags:
                del self.unattended_bags[bag_id]
                if bag_id in self.alerts_logged:
                    self.alerts_logged.remove(bag_id)
                    logger.info(f"Bag {bag_id} left frame, removing from unattended list.")

    def trigger_alert(self, bag_id: int, person_id: int):
        alert_msg = f"🚨 ALERT: Unattended bag detected! Bag Original ID: {bag_id}, Last Owner ID: {person_id}"
        logger.warning(alert_msg)
        print(alert_msg)

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]):
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            if detection.class_id == self.person_class_id:
                color = (0, 255, 0)
                display_id = detection.assigned_person_id
            elif detection.track_id in self.unattended_bags:
                color = (0, 0, 255)
                display_id = detection.assigned_person_id
            else:
                color = (255, 0, 0)
                display_id = detection.assigned_person_id
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if detection.class_id == self.person_class_id:
                label = f"{detection.class_name} ID:{display_id}"
            else:
                label = f"{detection.class_name} OwnerID:{display_id}"
                if detection.track_id in self.unattended_bags:
                    frames_unattended = self.frame_count - self.unattended_bags[detection.track_id]
                    label += f" UNATTENDED:{frames_unattended}"
                label += f" (YOLO:{detection.track_id})"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def draw_associations(self, frame: np.ndarray, detections: List[Detection]):
        person_detections = {d.track_id: d for d in detections if d.class_id == self.person_class_id}
        bag_detections = {d.track_id: d for d in detections if d.class_id in self.bag_class_ids}
        for bag_id, person_id in self.bag_id_mapping.items():
            if bag_id in bag_detections and person_id in person_detections and bag_id != person_id:
                person_center = person_detections[person_id].center
                bag_center = bag_detections[bag_id].center
                cv2.line(frame, person_center, bag_center, (255, 255, 0), 2)
                cv2.circle(frame, person_center, 8, (255, 255, 0), -1)
                cv2.circle(frame, bag_center, 8, (255, 255, 0), -1)

    def draw_info_panel(self, frame: np.ndarray):
        info_y = 30
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, info_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 30
        cv2.putText(frame, f"Active Persons: {len(self.active_persons)}", (10, info_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 30
        cv2.putText(frame, f"Active Bags (YOLO ID): {len(self.active_bags)}", (10, info_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 30
        cv2.putText(frame, f"Bags with Assigned IDs: {len(self.bag_id_mapping)}", (10, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        info_y += 30
        cv2.putText(frame, f"Unattended Bags (YOLO ID): {len(self.unattended_bags)}", (10, info_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if self.unattended_bags:
            info_y += 30
            cv2.putText(frame, "🚨 ALERT: Unattended luggage detected!", (10, info_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def process_video(self, input_source, output_path=None, display=True):
        if isinstance(input_source, str) and input_source.isdigit():
            cap = cv2.VideoCapture(int(input_source))
        else:
            cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            logger.error(f"Error opening video source: {input_source}")
            return
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Processing video: {width}x{height} @ {fps} FPS")
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self.frame_count += 1
                detections = self.detect_and_track(frame)
                self.update_associations(detections)
                self.check_unattended_bags()
                self.trajectory_tracker.draw_trails(frame)
                self.draw_detections(frame, detections)
                self.draw_associations(frame, detections)
                self.draw_info_panel(frame)
                if out:
                    out.write(frame)
                if display:
                    cv2.imshow('Person-Bag Tracking', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                if self.frame_count % 100 == 0:
                    logger.info(f"Processed {self.frame_count} frames")
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            logger.info(f"Video processing completed. Total frames: {self.frame_count}")

def main():
    parser = argparse.ArgumentParser(description='Person-Bag Tracking with Unattended Luggage Detection')
    parser.add_argument('--input', type=str, default='0', 
                        help='Input video file path or camera index (default: 0 for webcam)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video file path (optional)')
    parser.add_argument('--model', type=str, default='yolov8x.pt',
                        help='YOLOv8 model path (default: yolov8x.pt)')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable real-time display')
    args = parser.parse_args()
    tracker = PersonBagTracker(model_path=args.model)
    tracker.process_video(
        input_source=args.input,
        output_path=args.output,
        display=not args.no_display
    )

if __name__ == "__main__":
    main()
