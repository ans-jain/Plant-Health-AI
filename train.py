"""
Training Pipeline for PlantHealth-AI (MobileNetV2 Deep Vision Architecture)
Evaluates both Fine-Grained (6-Class) and Binary (Healthy vs. Diseased) Performance.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model_architecture import build_plant_classifier_model, DISEASE_KNOWLEDGE_BASE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12
SEED = 42

def train():
    print("=" * 70)
    print("PLANT HEALTH-AI — TRAINING MOBILENETV2 DEEP VISION MODEL")
    print("=" * 70)
    
    # 1. Load Dataset
    print(f"Loading dataset from: {DATASET_DIR}")
    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.25,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )
    
    raw_val_test_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.25,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )
    
    class_names = raw_train_ds.class_names
    num_classes = len(class_names)
    print(f"\nDetected {num_classes} classes: {class_names}")
    
    # Split val_test into 50% validation and 50% test
    val_test_cardinality = tf.data.experimental.cardinality(raw_val_test_ds).numpy()
    val_size = val_test_cardinality // 2
    val_ds = raw_val_test_ds.take(val_size)
    test_ds = raw_val_test_ds.skip(val_size)
    
    # 2. Data Augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1)
    ], name="data_augmentation")
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = raw_train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # 3. Model Construction
    print("\nInitializing Pretrained MobileNetV2 Backbone...")
    model = build_plant_classifier_model(num_classes=num_classes, input_shape=(224, 224, 3), fine_tune_layers=40)
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=2e-4,
        decay_steps=EPOCHS * len(raw_train_ds),
        alpha=0.05
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # 4. Callbacks
    model_save_path = os.path.join(MODELS_DIR, "plant_health_mobilenetv2.keras")
    cb_list = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(model_save_path, monitor="val_accuracy", save_best_only=True, verbose=1)
    ]
    
    # 5. Fit Model
    print("\nStarting Transfer Learning Optimization...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cb_list
    )
    
    # 6. Evaluation on Unseen Test Dataset
    print("\n" + "=" * 70)
    print("COMPREHENSIVE TEST SET EVALUATION")
    print("=" * 70)
    
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Multi-Class Metrics
    acc_multi = accuracy_score(y_true, y_pred)
    prec_multi, rec_multi, f1_multi, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    print(f"\n[FINE-GRAINED 6-CLASS METRICS]")
    print(f"Accuracy:  {acc_multi * 100:.2f}%")
    print(f"Precision: {prec_multi * 100:.2f}%")
    print(f"Recall:    {rec_multi * 100:.2f}%")
    print(f"F1-Score:  {f1_multi * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    # Binary Metrics (Healthy vs. Diseased)
    healthy_idx = class_names.index("Tomato_Healthy") if "Tomato_Healthy" in class_names else 0
    y_true_binary = np.array([0 if y == healthy_idx else 1 for y in y_true])
    y_pred_binary = np.array([0 if y == healthy_idx else 1 for y in y_pred])
    
    acc_bin = accuracy_score(y_true_binary, y_pred_binary)
    prec_bin, rec_bin, f1_bin, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average="binary", zero_division=0)
    
    print("\n" + "-" * 70)
    print("[BINARY CLASSIFICATION: HEALTHY (0) vs. DISEASED (1)]")
    print(f"Binary Accuracy:       {acc_bin * 100:.2f}%")
    print(f"Disease Recall (Sens): {rec_bin * 100:.2f}%")
    print(f"Precision:             {prec_bin * 100:.2f}%")
    print(f"F1-Score:              {f1_bin * 100:.2f}%")
    print("-" * 70)
    
    cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
    print("Binary Confusion Matrix (Rows=True [Healthy, Diseased], Cols=Pred [Healthy, Diseased]):")
    print(cm_binary)
    
    # Save Confusion Matrix Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Multi-class CM
    cm_multi = confusion_matrix(y_true, y_pred)
    im1 = ax1.imshow(cm_multi, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title('6-Class Confusion Matrix')
    fig.colorbar(im1, ax=ax1)
    ticks = np.arange(len(class_names))
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(class_names, fontsize=8)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # Binary CM
    im2 = ax2.imshow(cm_binary, interpolation='nearest', cmap=plt.cm.Greens)
    ax2.set_title('Binary Confusion Matrix (Healthy vs. Diseased)')
    fig.colorbar(im2, ax=ax2)
    bin_labels = ['Healthy', 'Diseased']
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(bin_labels)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(bin_labels)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm_binary[i, j]), ha="center", va="center", color="black", fontsize=14, fontweight="bold")
            
    plt.tight_layout()
    cm_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nSaved confusion matrix charts to: {cm_path}")
    
    # 7. Metadata and Knowledge Base
    metadata = {
        "class_names": class_names,
        "class_to_idx": {name: i for i, name in enumerate(class_names)},
        "metrics": {
            "binary_accuracy": float(acc_bin),
            "binary_disease_recall": float(rec_bin),
            "binary_precision": float(prec_bin),
            "binary_f1_score": float(f1_bin),
            "multiclass_accuracy": float(acc_multi),
            "multiclass_f1": float(f1_multi)
        },
        "knowledge_base": DISEASE_KNOWLEDGE_BASE
    }
    
    metadata_path = os.path.join(MODELS_DIR, "class_indices.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Model metadata saved to: {metadata_path}")
    print(f"Model saved to: {model_save_path}")
    print("=" * 70)
    print("TRAINING PROCESS COMPLETED!")
    print("=" * 70)

if __name__ == "__main__":
    train()
