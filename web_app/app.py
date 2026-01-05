from flask import Flask, render_template, request, jsonify
import os
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load model - assume best.pt is in the parent directory or standard run path
# You might need to adjust this path after training
MODEL_PATH = r'runs\detect\train2\weights\best.pt'
# Fallback to standard yolov8n for demo if training hasn't run yet
if not os.path.exists(MODEL_PATH):
    print(f"Warning: {MODEL_PATH} not found. Using yolov8n.pt for demonstration.")
    model = YOLO('yolov8n.pt')
else:
    model = YOLO(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Inference
    results = model(img)
    
    # Plot results
    res_plotted = results[0].plot()
    
    # Convert to base64 for display
    _, buffer = cv2.imencode('.jpg', res_plotted)
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({'image': img_str, 'detections': str(results[0].boxes)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
