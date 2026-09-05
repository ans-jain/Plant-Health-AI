"""
PlantHealth-AI: MobileNetV2 Deep Vision Architecture & Grad-CAM Explainability Engine
Designed for fast, high-accuracy plant pathology detection with explainable visual attention.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
import cv2
from PIL import Image

# Comprehensive Agronomic Pathology & Treatment Knowledge Base
DISEASE_KNOWLEDGE_BASE = {
    "Tomato_Healthy": {
        "status": "Healthy",
        "severity": "None",
        "scientific_name": "Solanum lycopersicum",
        "description": "Leaf exhibits uniform chlorophyll distribution, turgid cell structure, and no visible fungal spots or viral curling.",
        "cause": "Optimal growth conditions with balanced macro/micronutrients and appropriate irrigation.",
        "treatments": [
            "Maintain consistent drip irrigation at the root zone (avoid wetting leaves).",
            "Ensure proper crop spacing for air circulation and sunlight penetration.",
            "Continue periodic scouting for early pest or disease symptoms."
        ],
        "organic_care": "Apply compost tea or seaweed extract to strengthen natural plant immunity."
    },
    "Tomato_Early_Blight": {
        "status": "Diseased",
        "severity": "Moderate to High",
        "scientific_name": "Alternaria solani",
        "description": "Dark brown to black concentric circular lesions ('target-board' pattern) with chlorotic yellow halos, starting on older foliage.",
        "cause": "Fungal pathogen favored by warm temperatures (24-29°C) and prolonged leaf wetness.",
        "treatments": [
            "Apply protective copper-based fungicide (Copper Hydroxide) or Chlorothalonil upon first symptom onset.",
            "Prune infected lower foliage immediately and dispose away from the crop area.",
            "Avoid overhead sprinkler irrigation; switch to root-level drip irrigation."
        ],
        "organic_care": "Spray Bacillus subtilis (Serenade) or copper octanoate bio-fungicide."
    },
    "Tomato_Late_Blight": {
        "status": "Diseased",
        "severity": "Critical",
        "scientific_name": "Phytophthora infestans",
        "description": "Large, irregular water-soaked pale-green to dark brown lesions. Rapidly blights foliage and stems during humid conditions.",
        "cause": "Water mold (Oomycete) favored by cool, wet weather (<20°C and >90% relative humidity).",
        "treatments": [
            "Emergency application of systemic fungicides such as Mefenoxam or Mancozeb + Dimethomorph.",
            "Strictly isolate infected plants; remove and bag severely damaged vines immediately.",
            "Sterilize all pruning tools with 70% isopropyl alcohol between plants."
        ],
        "organic_care": "Preventive copper sulfate applications; ensure wide row spacing to maximize airflow."
    },
    "Tomato_Bacterial_Spot": {
        "status": "Diseased",
        "severity": "High",
        "scientific_name": "Xanthomonas perforans / euvesicatoria",
        "description": "Small, dark water-soaked spots (1-3mm) that turn angular and purplish-brown, often surrounded by yellow halos.",
        "cause": "Bacterial infection transmitted by seed, infected crop debris, and splashing rain.",
        "treatments": [
            "Spray copper bactericide combined with Mancozeb (significantly improves copper efficacy).",
            "Avoid handling or cultivating plants when foliage is wet to prevent bacterial spread.",
            "Practice minimum 2-year crop rotation with non-solanaceous crops."
        ],
        "organic_care": "Apply Actinovate (Streptomyces lydicus) or Agri-Phage bacteriophage spray."
    },
    "Tomato_Septoria_Leaf_Spot": {
        "status": "Diseased",
        "severity": "Moderate",
        "scientific_name": "Septoria lycopersici",
        "description": "Numerous small, circular lesions with dark brown margins and light tan or gray centers containing tiny black fungal pycnidia dots.",
        "cause": "Fungal spores overwintering in weeds and crop debris, dispersed by wind-driven rain.",
        "treatments": [
            "Apply Chlorothalonil, Pyraclostrobin, or Azoxystrobin fungicides at 7-10 day intervals.",
            "Mulch heavily around the base of plants to prevent soil-splash onto lower leaves.",
            "Prune bottom 12-18 inches of leaves to stop splash infection."
        ],
        "organic_care": "Neem oil or potassium bicarbonate sprays to suppress fungal sporulation."
    },
    "Tomato_Yellow_Leaf_Curl": {
        "status": "Diseased",
        "severity": "Severe",
        "scientific_name": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "description": "Upward curling and cupping of leaf margins, severe interveinal chlorosis (yellowing), stunting, and aborted fruit set.",
        "cause": "Begomovirus transmitted primarily by Silverleaf Whiteflies (Bemisia tabaci).",
        "treatments": [
            "Control whitefly vectors using systemic insecticides (Imidacloprid, Acetamiprid).",
            "Install insect-proof fine mesh netting (50-mesh) in greenhouses and nurseries.",
            "Rogue out and destroy infected viral plants immediately (viruses cannot be cured once inside vascular tissues)."
        ],
        "organic_care": "Use yellow sticky traps to monitor whiteflies; spray insecticidal soap or horticultural neem oil weekly."
    }
}

def build_plant_classifier_model(num_classes=6, input_shape=(224, 224, 3), fine_tune_layers=40):
    """
    Constructs a high-accuracy, lightweight deep learning model using MobileNetV2.
    Leverages ImageNet pre-trained feature extractors with fine-tuning.
    """
    # Base Backbone
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Enable training on upper layers for domain adaptation
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False
        
    # Classification Head
    inputs = layers.Input(shape=input_shape, name="leaf_image_input")
    x = layers.Rescaling(scale=1./127.5, offset=-1.0)(inputs)
    x = base_model(x, training=False)
    
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(
        128, 
        activation='relu', 
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_feature_projection"
    )(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name="disease_prediction")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="PlantHealth_MobileNetV2")
    return model

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1", pred_index=None):
    """
    Generates a Grad-CAM (Gradient-weighted Class Activation Mapping) heatmap
    to visualize the spatial features driving the model's prediction.
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break
            
    if base_model is None:
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
    else:
        rescaled_inputs = model.get_layer(index=1)(img_array)
        grad_model = tf.keras.models.Model(
            inputs=[base_model.inputs],
            outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, base_preds = grad_model(rescaled_inputs)
            x = model.get_layer("global_avg_pool")(base_preds)
            x = model.get_layer("batch_norm")(x)
            x = model.get_layer("dropout_1")(x)
            x = model.get_layer("dense_feature_projection")(x)
            x = model.get_layer("dropout_2")(x)
            predictions = model.get_layer("disease_prediction")(x)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def generate_gradcam_overlay(orig_img_path, heatmap, alpha=0.4):
    """
    Overlays the Grad-CAM heatmap onto the original image using OpenCV JET colormap.
    Returns the blended RGB image array.
    """
    img = cv2.imread(orig_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (img.shape[1], img.shape[0]))
    
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    superimposed_img = heatmap_colored * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img
