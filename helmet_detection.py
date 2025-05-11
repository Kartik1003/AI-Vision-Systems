import cv2
import threading
from ultralytics import YOLO
import time

class HelmetController:
    def __init__(self, model_path, source1=0):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(source1)
        self.count_no_helmet = 0
        self.lock = threading.Lock()
        self.signal = 'OK'
        self.remaining = 0
        self.last_check_time = time.time()

    def generate_frames(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                # Reset video to beginning if it ends
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = self.cap.read()
                if not success:
                    break

            # Resize frame for consistency
            frame = cv2.resize(frame, (640, 360))
            
            # Run detection
            results = self.model(frame)[0]
            no_helmet_count = 0
            
            # Draw bounding boxes and labels
            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                if int(cls) == 1:  # No helmet class (index 1)
                    no_helmet_count += 1
                    color, label = (0, 0, 255), 'Helmet'
                else:  # Helmet class (index 0)
                    color, label = (0, 255, 0), 'No Helmet'
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add status overlay
            cv2.putText(frame, f"No Helmet Count: {no_helmet_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Status: {self.signal}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            with self.lock:
                self.count_no_helmet = no_helmet_count
            
            # Check time-based logic
            current_time = time.time()
            if current_time - self.last_check_time >= 1:  # Update every second
                self.last_check_time = current_time
                self.update_signal_status()
                
            # Stream the frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def update_signal_status(self):
        with self.lock:
            count = self.count_no_helmet
        
        if self.remaining > 0:
            self.remaining -= 1
        
        if count > 0:
            self.signal = 'ALERT'
            self.remaining = 10
        elif self.remaining <= 0:
            self.signal = 'OK'

    def get_status(self):
        with self.lock:
            count = self.count_no_helmet
        return {
            'count': count,
            'signal': self.signal,
            'remaining': self.remaining
        }
