from flask import Flask, render_template, Response, jsonify, request
from traffic_control import TrafficController
from helmet_detection import HelmetController
import threading

app = Flask(__name__)

# Initialize controllers
traffic_controller = TrafficController(
    model_path="yolov8n.pt",
    source1="Footage1.mp4",
    source2="Footage2.mp4"
)

helmet_controller = HelmetController(
    model_path="yolov8n.pt", 
    source1="footage3.mp4"
)

# Track active mode
active_mode = "traffic"  # Default to traffic control mode

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Route video feed based on active mode
    if active_mode == "traffic":
        return Response(
            traffic_controller.generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    else:
        return Response(
            helmet_controller.generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

@app.route('/status')
def status():
    # Return status based on active mode
    if active_mode == "traffic":
        stats = traffic_controller.get_status()
        return jsonify({
            'mode': 'traffic',
            'count': stats['count'],
            'timer': int(stats['remaining']),
            'signal': stats['signal']
        })
    else:
        stats = helmet_controller.get_status()
        return jsonify({
            'mode': 'helmet',
            'no_helmet_count': stats['count'],
            'timer': int(stats['remaining']),
            'signal': stats['signal']
        })

@app.route('/switch_mode', methods=['POST'])
def switch_mode():
    global active_mode
    new_mode = request.json.get('mode')
    if new_mode in ["traffic", "helmet"]:
        active_mode = new_mode
        return jsonify({"success": True, "mode": active_mode})
    return jsonify({"success": False, "error": "Invalid mode"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)