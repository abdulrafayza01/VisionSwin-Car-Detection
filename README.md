# YOLOv8 Swin Car Detection Project

This project contains a pipeline to train a YOLOv8 model on the Car dataset and an interactive 3D web interface for inference.

## Project Structure
- `train_model.ipynb`: Jupyter notebook to train the model.
- `web_app/`: Flask application.
    - `app.py`: Backend server.
    - `templates/index.html`: Frontend HTML.
    - `static/`: assets (JS/CSS).
- `Car.v1i.yolov8/`: Dataset.

## Quick Start

### 1. Train
Run the `train_model.ipynb` notebook.
This will generate `runs/detect/train/weights/best.pt`.

### 2. Run Interface
```powershell
cd web_app
python app.py
```
Visit `localhost:5000` to see the 3D interface.
