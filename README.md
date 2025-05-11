# AI Vision Systems

**Real-time Traffic Signal Control & Helmet Detection**  
Built with YOLOv8 and Flask during the **Galgotia International Hackathon 2025** (May 9–10).

---

## 🚀 Project Overview
AI Vision Systems is a lightweight web application that seamlessly switches between two real-time video-analysis modes:
1. **Traffic Control**  
   • Counts vehicles on two camera feeds  
   • Dynamically adjusts traffic-signal timing  
   • Logs traffic statistics  
2. **Helmet Detection**  
   • Monitors a live feed or video source  
   • Counts instances of riders without helmets  
   • Raises real-time safety alerts  

All processing runs on YOLOv8 models and OpenCV, with a Flask front-end that displays video streams, stats panels, and mode-switch controls.

---

## ⚙️ Tech Stack
- **Backend**: Python · Flask  
- **CV & AI**: Ultralytics YOLOv8 · OpenCV  
- **Frontend**: Bootstrap 5 · HTML/CSS/JS  
- **Model**: `yolov8n.pt`  
- **Sample Feeds**: `Footage1.mp4`, `Footage2.mp4`, `footage3.mp4`

---

## 📦 Installation

1. Clone this repo  
   ```bash
   git clone https://github.com/your-org/ai-vision-systems.git
   cd ai-vision-systems
