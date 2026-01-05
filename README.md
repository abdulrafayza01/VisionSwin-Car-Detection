# 🏎️ VisionSwin: Hybrid YOLOv8 + Swin Transformer Car Detection

VisionSwin is a next-generation object detection system that bridges the gap between **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViT)**. By integrating a Swin Transformer backbone into the YOLOv8 architecture, this project achieves superior feature extraction, especially in complex environments.

![Project Preview](web_app\static\Background.png)

## 🚀 Why Swin Transformer?
Standard YOLOv8 uses CSPDarknet (CNN), which is great for local features. However, **Swin Transformer** uses hierarchical window-based self-attention, allowing the model to:
- Understand **Global Context** (how objects relate to their surroundings).
- Handle **Occlusions** (detecting cars even if they are partially hidden).
- Scale efficiently across different image resolutions.



## ✨ Features
- **3D Immersive UI:** A futuristic interface built with **Three.js** featuring real-time particle physics and 3D Torus-Knot animations.
- **Hybrid Architecture:** Custom-modified YOLOv8n with a Swin-Tiny backbone via `timm` integration.
- **Real-Time Inference:** Fast Flask backend that processes images and returns base64-encoded results with bounding boxes.
- **Glassmorphism Design:** Modern UI with blur effects and CSS interactive gradients.

## 🛠️ Tech Stack
- **Deep Learning:** PyTorch, Ultralytics (YOLOv8), TIMM (Swin Transformer).
- **Backend:** Flask (Python).
- **Frontend:** Three.js (3D), Vanilla JavaScript, CSS3.
- **Data Science:** OpenCV, NumPy, Matplotlib.

## ⚙️ Installation & Setup

1. **Clone the Repo:**
   ```bash```
   git clone [https://github.com/abdulrafayza01/VisionSwin-Car-Detection.git](https://github.com/abdulrafayza01/VisionSwin-Car-Detection.git)
   cd VisionSwin-Car-Detection

2. **Install Dependencies:**

    pip install -r requirements.txt

3. **Model Weights:**

    Ensure your trained best.pt is located at runs/detect/train2/weights/best.pt. (The repo will fallback to yolov8n.pt if not found).

4. **Run Server:**

    python app.py
    Access the app at http://127.0.0.1:5000.

🧠 Architecture Logic
The system uses a custom SwinBackbone class that returns multi-scale features (P3, P4, P5). These are then routed into the YOLOv8 Head using a specialized ListSelector module, ensuring that the transformer's rich spatial features are utilized for precise bounding box regression.

Developed with ❤️ by abdulrafayza01

