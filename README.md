# PlantHealth-AI: Explainable Deep Vision Plant Pathology Diagnostic System

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end, explainable agricultural computer vision platform that diagnoses plant foliage diseases from raw leaf photographs in real-time. Built using **MobileNetV2 Deep Transfer Learning**, **Grad-CAM (Gradient-weighted Class Activation Mapping)** for visual explainability, and an **Agronomic Pathology & Treatment Engine** served via a **Flask REST API & Web Dashboard**.

---

## 📸 Key Features

- **⚡ Real-Time Edge Inference**: Achieves **15–25 ms latency per image** on standard commodity CPUs (~50+ FPS throughput).
- **🔬 Explainable AI (Grad-CAM)**: Generates spatial attention heatmaps showing the exact leaf lesions, necrotic spots, and chlorotic halos that triggered the diagnosis.
- **🎯 Dual-Level Diagnostics**:
  - **Binary Health Status**: Distinguishes **Healthy** vs. **Diseased** leaves (88.1% Accuracy, 97.0% Precision, 92.9% F1).
  - **Fine-Grained Disease Identification**: Categorizes specific fungal, bacterial, and viral pathologies (Early Blight, Late Blight, Bacterial Spot, Septoria Leaf Spot, Yellow Leaf Curl).
- **🌱 Actionable Agronomic Prescriptions**: Instantly recommends protective chemical fungicides, organic bio-controls (e.g., *Bacillus subtilis*, neem formulations), and cultural management practices.
- **💻 Interactive Web Dashboard & REST API**: Drag-and-drop web UI with dynamic Grad-CAM side-by-side visualization and clean JSON API endpoints.

---

## 🏗️ System Architecture

```
[Raw Leaf Image Input]
        │
        ▼
[Bilinear Resize to 224×224 RGB, [-1, 1] Normalization]
        │
        ▼
[MobileNetV2 Pretrained Convolutional Backbone (ImageNet)]
        │
        ▼
[Global Average Pooling 2D + Batch Normalization + Dropout (0.3)]
        │
        ▼
[Dense Feature Projection (128 units, L2 Regularization, ReLU)]
        │
        ▼
[Softmax Classification Head] ──► [Predicted Condition & Confidence Score]
        │
        ├──► [Grad-CAM Engine (Conv_1 Feature Maps)] ──► [Heatmap Overlay]
        └──► [Pathology Knowledge Base] ─────────────► [Agronomic Treatment Plan]
```

---

## 📊 Evaluation & Performance Metrics

Trained on a balanced subset of the gold-standard **PlantVillage** dataset across 6 pathology categories:

```text
----------------------------------------------------------------------
BINARY HEALTH STATUS METRICS (Healthy vs. Diseased)
----------------------------------------------------------------------
• Binary Classification Accuracy:  88.10%
• Precision:                       97.01%
• F1-Score:                        92.86%
• Disease Sensitivity (Recall):    89.04%
----------------------------------------------------------------------
INFERENCE PERFORMANCE (Intel Core / AMD CPU, Batch Size = 1)
----------------------------------------------------------------------
• Mean Latency:                    19.4 ms
• 95th Percentile Latency (p95):   24.1 ms
• Throughput:                      ~51.5 FPS
• Model Disk Footprint:            13.8 MB
----------------------------------------------------------------------
```

---

## 🛠️ Technology Stack & Libraries Used

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Deep Learning** | `TensorFlow` / `Keras` | Neural network construction, MobileNetV2 pretrained backbone, fine-tuning. |
| **Explainable AI (XAI)** | `tf.GradientTape` + `OpenCV` | Computing feature gradients and superimposing JET colormap Grad-CAM heatmaps. |
| **Image Processing** | `Pillow` (`PIL`), `OpenCV` | Image decoding, format standardization (RGB), and array transformations. |
| **Classical ML & Metrics** | `scikit-learn`, `NumPy` | Stratified train/val/test splits, Confusion Matrix, Precision-Recall-F1 evaluation. |
| **Visualization** | `matplotlib` | Generating dual-matrix evaluation charts. |
| **Web & API Backend** | `Flask`, `Flask-CORS` | Routing, asynchronous multipart upload handling, JSON API serving. |
| **Frontend** | HTML5, CSS3, Modern JS | Responsive glassmorphism dashboard, drag-and-drop file upload. |

---

## 📂 Repository Structure

```
PlantHealth-AI/
├── app.py                     # Flask web server and REST API
├── model_architecture.py      # MobileNetV2 architecture, Grad-CAM generator, and knowledge base
├── download_dataset.py        # Automated dataset fetcher from PlantVillage repository
├── train.py                   # Transfer learning pipeline & dual-metric evaluation
├── benchmark_evaluator.py     # Production latency & throughput benchmarking script
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Standard git exclusions
├── README.md                  # Project documentation & benchmark overview
├── models/
│   ├── plant_health_mobilenetv2.keras  # Pre-trained deep vision model weights
│   ├── class_indices.json              # Class mapping, performance metrics, and prescriptions
│   └── confusion_matrix.png            # Multi-class and binary confusion matrix charts
├── static/
│   ├── css/style.css          # Responsive dark-theme styling
│   └── js/app.js              # Client-side asynchronous upload & Grad-CAM viewer
└── templates/
    └── index.html             # Web dashboard interface
```

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/PlantHealth-AI.git
cd PlantHealth-AI
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
python app.py
```
Open your browser and navigate to:
```text
http://localhost:5002
```

### 3. (Optional) Run Performance Benchmark
```bash
python benchmark_evaluator.py
```

### 4. (Optional) Re-train or Fine-Tune Model
```bash
python download_dataset.py
python train.py
```

---

## 📡 REST API Endpoints

### `POST /api/classify`
Receives a leaf image file (`multipart/form-data`) and returns a complete diagnostic report.

**Response Example:**
```json
{
  "status": "success",
  "health_status": "Diseased",
  "predicted_class": "Tomato_Early_Blight",
  "confidence": 0.942,
  "details": {
    "scientific_name": "Alternaria solani",
    "severity": "Moderate to High",
    "cause": "Fungal pathogen favored by warm temperatures (24-29°C) and leaf wetness.",
    "treatments": [
      "Apply protective copper-based fungicide (Copper Hydroxide) or Chlorothalonil.",
      "Prune infected lower foliage immediately.",
      "Switch to drip irrigation."
    ],
    "organic_care": "Spray Bacillus subtilis (Serenade) bio-fungicide."
  },
  "original_image_base64": "data:image/jpeg;base64,...",
  "gradcam_base64": "data:image/jpeg;base64,..."
}
```

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).

## Author

**Anshika**

