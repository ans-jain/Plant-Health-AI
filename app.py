"""
PlantHealth-AI Pro — Flask Web Application & REST API
High-accuracy Deep Transfer Learning (MobileNetV2) with Grad-CAM Explainability.
"""

import os
import io
import json
import base64
import numpy as np
from PIL import Image
import cv2
from flask import Flask, render_template, request, jsonify
import tensorflow as tf

from model_architecture import (
    DISEASE_KNOWLEDGE_BASE, 
    make_gradcam_heatmap, 
    generate_gradcam_overlay
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB limit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "plant_health_mobilenetv2.keras")
METADATA_PATH = os.path.join(MODELS_DIR, "class_indices.json")

# Global model and metadata state
model = None
metadata = None
class_names = []

def load_model_and_metadata():
    global model, metadata, class_names
    if os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH):
        try:
            print("Loading trained MobileNetV2 model...")
            model = tf.keras.models.load_model(MODEL_PATH)
            with open(METADATA_PATH, "r") as f:
                metadata = json.load(f)
            class_names = metadata.get("class_names", [])
            print(f"Model loaded successfully with {len(class_names)} classes: {class_names}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    else:
        print("Model file not found. Please run train.py first.")
        return False

# Initialize model at startup
load_model_and_metadata()

@app.route('/')
def index():
    """Serves the main AI diagnostic dashboard"""
    return render_template('index.html', model_ready=(model is not None))

@app.route('/api/info')
def api_info():
    """Returns model metadata and performance metrics"""
    if metadata:
        return jsonify({
            'status': 'ready',
            'model_name': 'MobileNetV2 Deep Vision Classifier',
            'classes': class_names,
            'metrics': metadata.get('metrics', {})
        })
    return jsonify({'status': 'not_ready', 'message': 'Model not yet trained.'})

@app.route('/api/classify', methods=['POST'])
def classify():
    """Real-time leaf image diagnosis with Grad-CAM visual explanation"""
    global model, class_names
    
    if model is None:
        # Attempt lazy reload
        if not load_model_and_metadata():
            return jsonify({
                'status': 'error', 
                'message': 'Model is not loaded. Please run train.py to generate the trained model.'
            }), 500

    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image file uploaded.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'Empty file selected.'}), 400

    try:
        # 1. Read and preprocess image
        image_bytes = file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Save temporary file for Grad-CAM overlay processing
        temp_dir = os.path.join(BASE_DIR, "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"temp_{os.getpid()}_{file.filename}")
        pil_img.save(temp_path)

        # Format input tensor for MobileNetV2 (224x224x3)
        resized_img = pil_img.resize((224, 224))
        img_array = np.array(resized_img, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)

        # 2. Forward Inference
        predictions = model.predict(img_batch, verbose=0)[0]
        top_idx = int(np.argmax(predictions))
        confidence = float(predictions[top_idx])
        predicted_class = class_names[top_idx] if top_idx < len(class_names) else "Unknown"

        # 3. Determine Binary Health Status
        is_healthy = (predicted_class == "Tomato_Healthy")
        health_status = "Healthy" if is_healthy else "Diseased"

        # 4. Generate Grad-CAM Heatmap
        try:
            heatmap = make_gradcam_heatmap(img_batch, model, last_conv_layer_name="Conv_1", pred_index=top_idx)
            gradcam_img = generate_gradcam_overlay(temp_path, heatmap, alpha=0.4)
        except Exception as cam_err:
            print(f"Grad-CAM generation fallback: {cam_err}")
            # Fallback: display original image
            gradcam_img = np.array(resized_img)

        # 5. Convert images to Base64 strings for instant UI rendering
        # Original Image Base64
        orig_buffer = io.BytesIO()
        pil_img.save(orig_buffer, format="JPEG", quality=85)
        orig_base64 = "data:image/jpeg;base64," + base64.b64encode(orig_buffer.getvalue()).decode('utf-8')

        # Grad-CAM Overlay Base64
        gradcam_pil = Image.fromarray(gradcam_img)
        cam_buffer = io.BytesIO()
        gradcam_pil.save(cam_buffer, format="JPEG", quality=85)
        cam_base64 = "data:image/jpeg;base64," + base64.b64encode(cam_buffer.getvalue()).decode('utf-8')

        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # 6. Retrieve Agronomic Pathology & Prescriptions
        disease_info = DISEASE_KNOWLEDGE_BASE.get(predicted_class, {
            "status": health_status,
            "severity": "Unknown",
            "cause": "Consult a certified plant pathologist.",
            "treatments": ["Isolate plant", "Monitor soil moisture"],
            "organic_care": "Apply balanced organic fertilizer."
        })

        return jsonify({
            'status': 'success',
            'health_status': health_status,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_probabilities': {
                name: float(prob) for name, prob in zip(class_names, predictions)
            },
            'details': disease_info,
            'original_image_base64': orig_base64,
            'gradcam_base64': cam_base64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f"Inference error: {str(e)}"}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Starting PlantHealth-AI Pro Server on http://0.0.0.0:5002")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5002)
