import cv2
import time
import numpy as np
from ultralytics import YOLO

class TrafficController:
    def __init__(self, model_path, source1, source2, output_video_prefix="output_log"):
        self.model = YOLO(model_path)
        self.cam1 = cv2.VideoCapture(source1)
        self.cam2 = cv2.VideoCapture(source2)

        # Strictly allowed vehicle classes
        self.vehicle_classes = ["car", "truck", "bus", "motorbike", "ambulance"]

        self.current_green = "A"
        self.previous_green = "A"
        self.last_state_change = time.time()
        self.green_duration = 5
        self.yellow_duration = 2
        self.in_yellow_phase = False

        self.count1 = 0
        self.count2 = 0
        self.remaining = 0

        # Generate unique output video name with custom prefix
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_video_path = f"{output_video_prefix}_{timestamp}.avi"
        self.out = None
        
        # Handle cases where videos reach the end
        self.reset_videos = True

    def count_vehicles(self, results):
        count = 0
        ambulance_found = False

        # Keywords indicating plant-like classes to skip
        plant_keywords = ["plant", "tree", "flower", "pot"]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            name = self.model.names.get(cls_id, "unknown").lower()

            # Ignore detections with plant-like class names
            if any(keyword in name for keyword in plant_keywords):
                continue

            if name == "ambulance":
                ambulance_found = True

            if name in self.vehicle_classes:
                count += 1

        return count, ambulance_found

    def draw_signal_panel(self, is_current, is_yellow, duration, width=100, height=360):
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(panel, (20, 80), (80, 280), (100, 100, 100), -1)
        lights = {"green": (50, 100), "yellow": (50, 180), "red": (50, 260)}
        active = "yellow" if is_yellow else "green" if is_current else "red"
        for color, pos in lights.items():
            light_color = (50, 50, 50)
            if color == active:
                light_color = {
                    "green": (0, 255, 0),
                    "yellow": (0, 255, 255),
                    "red": (0, 0, 255)
                }[color]
            cv2.circle(panel, pos, 20, light_color, -1)
        cv2.putText(panel, f"{int(duration)}s", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return panel

    def generate_frames(self):
        while True:
            ret1, frame1 = self.cam1.read()
            ret2, frame2 = self.cam2.read()
            
            # Handle video ending - reset to beginning
            if not ret1 or not ret2:
                if self.reset_videos:
                    self.cam1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.cam2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret1, frame1 = self.cam1.read()
                    ret2, frame2 = self.cam2.read()
                    if not ret1 or not ret2:
                        break
                else:
                    break

            frame1 = cv2.resize(frame1, (640, 360))
            frame2 = cv2.resize(frame2, (640, 360))

            res1 = self.model(frame1)[0]
            res2 = self.model(frame2)[0]

            self.count1, amb1 = self.count_vehicles(res1)
            self.count2, amb2 = self.count_vehicles(res2)
            ambulance_detected = amb1 or amb2

            now = time.time()
            elapsed = now - self.last_state_change

            if ambulance_detected:
                self.current_green = "A" if amb1 else "B"
                self.green_duration = 10
                self.last_state_change = now
                self.in_yellow_phase = False
            elif not self.in_yellow_phase and elapsed >= self.green_duration:
                self.current_green = "YELLOW"
                self.in_yellow_phase = True
                self.last_state_change = now
            elif self.in_yellow_phase and elapsed >= self.yellow_duration:
                self.current_green = "B" if self.previous_green == "A" else "A"
                self.green_duration = min(10, 3 + (self.count2 if self.current_green == "B" else self.count1) // 2)
                self.previous_green = self.current_green
                self.last_state_change = now
                self.in_yellow_phase = False

            self.remaining = ((self.yellow_duration if self.current_green == "YELLOW" else self.green_duration)
                              - (time.time() - self.last_state_change))

            # Filter out plant-like detections before plotting
            def filter_boxes(results, model, plant_keywords):
                return results.boxes[
                    [not any(kw in model.names[int(b.cls[0])].lower() for kw in plant_keywords)
                     for b in results.boxes]
                ]

            plant_keywords = ["plant", "tree", "flower", "pot"]
            res1.boxes = filter_boxes(res1, self.model, plant_keywords)
            res2.boxes = filter_boxes(res2, self.model, plant_keywords)

            ann1 = res1.plot()
            ann2 = res2.plot()

            cv2.putText(ann1, f"Vehicles: {self.count1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(ann2, f"Vehicles: {self.count2}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            panel1 = self.draw_signal_panel(self.current_green == "A", self.current_green == "YELLOW" and self.previous_green == "A", self.remaining)
            panel2 = self.draw_signal_panel(self.current_green == "B", self.current_green == "YELLOW" and self.previous_green == "B", self.remaining)

            stacked = np.vstack((np.hstack((ann1, panel1)), np.hstack((ann2, panel2))))

            if self.out is None:
                height, width, _ = stacked.shape
                self.out = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))

            self.out.write(stacked)

            ret, buffer = cv2.imencode('.jpg', stacked)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        self.cam1.release()
        self.cam2.release()
        if self.out:
            self.out.release()

    def get_status(self):
        current_count = self.count1 if self.current_green == "A" else self.count2
        signal_state = 'yellow' if self.current_green == 'YELLOW' else ('green' if self.current_green == 'A' else 'red')
        
        # For "B" road, we need to reverse the signal state for the frontend
        if self.current_green == "B":
            signal_state = "green"
        
        return {
            'count': current_count,
            'remaining': max(0, self.remaining),
            'signal': signal_state
        }