"""
LD7182 - AI for IoT
Smart Agriculture & Crop Recommendation - Full ML Pipeline
==========================================================
Steps:
  1. Load Crop_recommendation.csv
  2. Preprocess (encode labels, normalise features)
  3. Train Keras Neural Network
  4. Evaluate (accuracy, precision, recall, F1, confusion matrix)
  5. Convert to TensorFlow Lite (float32) AND TensorFlow Lite (INT8 quantised)
  6. Benchmark TFLite inference (size + latency)
  7. Export C-header file ready for ESP32-S3 (TinyML deployment)
"""

import os, sys, time, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ------------------ Configuration ------------------
DATA_PATH   = '/mnt/user-data/uploads/Crop_recommendation.csv'
OUT_DIR     = '/home/claude/tech_outputs'
SEED        = 42
EPOCHS      = 80
BATCH_SIZE  = 32

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)
tf.random.set_seed(SEED)

LOG_FILE = open(os.path.join(OUT_DIR, 'training_log.txt'), 'w')
def log(msg=""):
    print(msg)
    LOG_FILE.write(str(msg) + "\n")
    LOG_FILE.flush()

log("="*70)
log("LD7182 - AI for IoT  |  ML Training Pipeline Execution Log")
log("="*70)
log(f"Python    : {sys.version.split()[0]}")
log(f"TensorFlow: {tf.__version__}")
log(f"NumPy     : {np.__version__}")
log(f"Random seed: {SEED}")
log("")

# ------------------ 1. Load Dataset ------------------
log(">> [1/7] Loading dataset ...")
df = pd.read_csv(DATA_PATH)
log(f"   Shape          : {df.shape}")
log(f"   Features       : {list(df.columns[:-1])}")
log(f"   Target classes : {df['label'].nunique()}")
log(f"   Missing values : {df.isnull().sum().sum()}")
log("")

# ------------------ 2. Preprocessing ------------------
log(">> [2/7] Preprocessing ...")
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])
NUM_CLASSES = df['label_enc'].nunique()
CLASS_NAMES = list(le.classes_)
log(f"   Encoded {NUM_CLASSES} crop classes")

X = df[['N','P','K','temperature','humidity','ph','rainfall']].values.astype(np.float32)
y = df['label_enc'].values.astype(np.int32)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=SEED, stratify=y)
log(f"   Train size: {X_train.shape}   Test size: {X_test.shape}")

# Save scaler params for ESP32-S3 firmware
scaler_params = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "feature_names": ['N','P','K','temperature','humidity','ph','rainfall'],
    "class_names": CLASS_NAMES
}
with open(os.path.join(OUT_DIR, 'scaler_params.json'), 'w') as f:
    json.dump(scaler_params, f, indent=2)
log("   Saved scaler_params.json")
log("")

# ------------------ 3. Train Keras Neural Network ------------------
log(">> [3/7] Building & training Keras NN (TinyML-compatible) ...")
model = keras.Sequential([
    keras.layers.Input(shape=(7,), name='soil_climate_input'),
    keras.layers.Dense(32, activation='relu', name='hidden1'),
    keras.layers.Dense(16, activation='relu', name='hidden2'),
    keras.layers.Dense(NUM_CLASSES, activation='softmax', name='crop_output')
])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# capture model summary
ms_lines = []
model.summary(print_fn=lambda s: ms_lines.append(s))
for ln in ms_lines: log("   " + ln)

t0 = time.time()
history = model.fit(X_train, y_train,
                    epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_split=0.10, verbose=0)
train_time = time.time() - t0
log(f"   Training completed in {train_time:.2f} s over {EPOCHS} epochs")
log("")

# Save training-curve plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(history.history['accuracy'], label='train', color='#2E7D32', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='val', color='#1565C0', linewidth=2)
axes[0].set_title('Model Accuracy Over Epochs', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(history.history['loss'], label='train', color='#C62828', linewidth=2)
axes[1].plot(history.history['val_loss'], label='val', color='#F57C00', linewidth=2)
axes[1].set_title('Model Loss Over Epochs', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'training_curves.png'), dpi=160, bbox_inches='tight')
plt.close()
log("   Saved training_curves.png")
log("")

# ------------------ 4. Evaluation ------------------
log(">> [4/7] Evaluating Keras model on test set ...")
preds_proba = model.predict(X_test, verbose=0)
preds = np.argmax(preds_proba, axis=1)

acc  = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, average='weighted', zero_division=0)
rec  = recall_score(y_test, preds, average='weighted', zero_division=0)
f1   = f1_score(y_test, preds, average='weighted', zero_division=0)
log(f"   Accuracy : {acc:.4f}")
log(f"   Precision: {prec:.4f}")
log(f"   Recall   : {rec:.4f}")
log(f"   F1-Score : {f1:.4f}")
log("")

with open(os.path.join(OUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(classification_report(y_test, preds, target_names=CLASS_NAMES))

# Confusion matrix for Keras NN
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(13, 11))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Predictions'})
plt.title(f'Confusion Matrix — Keras NN  (Test Accuracy = {acc*100:.2f}%)',
          fontsize=13, fontweight='bold')
plt.xlabel('Predicted Crop'); plt.ylabel('Actual Crop')
plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cm_keras_nn.png'), dpi=160, bbox_inches='tight')
plt.close()
log("   Saved cm_keras_nn.png and classification_report.txt")
log("")

# Save keras model
model.save(os.path.join(OUT_DIR, 'crop_model.keras'))

# ------------------ 5. TFLite conversion ------------------
log(">> [5/7] Converting model to TensorFlow Lite (TinyML) ...")

# --- Float32 TFLite (baseline) ---
converter_fp = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_fp = converter_fp.convert()
fp_path = os.path.join(OUT_DIR, 'crop_model_float32.tflite')
with open(fp_path, 'wb') as f: f.write(tflite_fp)
fp_size = len(tflite_fp)
log(f"   Float32 TFLite size: {fp_size:,} bytes ({fp_size/1024:.1f} KB)")

# --- INT8 quantised TFLite (for ESP32-S3 deployment) ---
def representative_dataset():
    for i in range(min(200, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.int8
converter_int8.inference_output_type = tf.int8
tflite_int8 = converter_int8.convert()
int8_path = os.path.join(OUT_DIR, 'crop_model_int8.tflite')
with open(int8_path, 'wb') as f: f.write(tflite_int8)
int8_size = len(tflite_int8)
log(f"   INT8    TFLite size: {int8_size:,} bytes ({int8_size/1024:.1f} KB)")
log(f"   Compression ratio: {fp_size/int8_size:.2f}x")
log("")

# ------------------ 6. TFLite Benchmarking ------------------
log(">> [6/7] Benchmarking TFLite models ...")

def run_tflite_eval(tflite_path, X, y, quantised=False):
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    if quantised:
        in_scale, in_zp = inp['quantization']
        out_scale, out_zp = out['quantization']

    preds = []
    latencies = []
    for i in range(len(X)):
        sample = X[i:i+1].astype(np.float32)
        if quantised:
            q_sample = (sample / in_scale + in_zp).astype(np.int8)
            interp.set_tensor(inp['index'], q_sample)
        else:
            interp.set_tensor(inp['index'], sample)
        t0 = time.perf_counter()
        interp.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)  # ms
        result = interp.get_tensor(out['index'])
        preds.append(np.argmax(result))
    preds = np.array(preds)
    a = accuracy_score(y, preds)
    return a, np.mean(latencies), np.median(latencies), np.max(latencies), preds

fp_acc, fp_mean_ms, fp_med_ms, fp_max_ms, fp_preds = run_tflite_eval(fp_path, X_test, y_test, False)
log(f"   Float32  -> Accuracy {fp_acc:.4f}  | mean {fp_mean_ms:.3f} ms | median {fp_med_ms:.3f} ms | max {fp_max_ms:.3f} ms")

int8_acc, int8_mean_ms, int8_med_ms, int8_max_ms, int8_preds = run_tflite_eval(int8_path, X_test, y_test, True)
log(f"   INT8     -> Accuracy {int8_acc:.4f}  | mean {int8_mean_ms:.3f} ms | median {int8_med_ms:.3f} ms | max {int8_max_ms:.3f} ms")
log("")

# Confusion matrix for INT8 (the actual ESP32 model)
cm_int8 = confusion_matrix(y_test, int8_preds)
plt.figure(figsize=(13, 11))
sns.heatmap(cm_int8, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f'Confusion Matrix — INT8 TFLite (Deployed on ESP32-S3)  Acc = {int8_acc*100:.2f}%',
          fontsize=13, fontweight='bold')
plt.xlabel('Predicted Crop'); plt.ylabel('Actual Crop')
plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cm_tflite_int8.png'), dpi=160, bbox_inches='tight')
plt.close()

# Benchmark comparison chart
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
sizes = [fp_size/1024, int8_size/1024]
labels = ['Float32 TFLite', 'INT8 TFLite\n(ESP32-S3)']
colors = ['#1976D2', '#2E7D32']
axes[0].bar(labels, sizes, color=colors, edgecolor='black')
for i, v in enumerate(sizes):
    axes[0].text(i, v + 0.5, f'{v:.1f} KB', ha='center', fontweight='bold')
axes[0].set_ylabel('Model Size (KB)')
axes[0].set_title('Model Size Comparison', fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

lats = [fp_mean_ms, int8_mean_ms]
axes[1].bar(labels, lats, color=colors, edgecolor='black')
for i, v in enumerate(lats):
    axes[1].text(i, v + 0.005, f'{v:.3f} ms', ha='center', fontweight='bold')
axes[1].set_ylabel('Mean Inference Latency (ms)')
axes[1].set_title('Inference Latency Comparison', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'tflite_benchmark.png'), dpi=160, bbox_inches='tight')
plt.close()
log("   Saved cm_tflite_int8.png and tflite_benchmark.png")
log("")

# ------------------ 7. Export C-header for ESP32-S3 ------------------
log(">> [7/7] Generating C header (crop_model_data.h) for ESP32-S3 ...")

def to_c_header(tflite_bytes, var_name, header_path):
    lines = []
    lines.append(f"// Auto-generated TFLite model array for ESP32-S3")
    lines.append(f"// Model size: {len(tflite_bytes):,} bytes")
    lines.append(f"#ifndef CROP_MODEL_DATA_H")
    lines.append(f"#define CROP_MODEL_DATA_H")
    lines.append(f"")
    lines.append(f"const unsigned int {var_name}_len = {len(tflite_bytes)};")
    lines.append(f"alignas(8) const unsigned char {var_name}[] = {{")
    chunk = []
    for i, b in enumerate(tflite_bytes):
        chunk.append(f"0x{b:02x}")
        if (i + 1) % 12 == 0:
            lines.append("  " + ", ".join(chunk) + ",")
            chunk = []
    if chunk:
        lines.append("  " + ", ".join(chunk))
    lines.append("};")
    lines.append("")
    lines.append("#endif")
    with open(header_path, 'w') as f:
        f.write("\n".join(lines))
    return header_path

header_path = to_c_header(tflite_int8, 'crop_model',
                          os.path.join(OUT_DIR, 'crop_model_data.h'))
log(f"   Header file written: {header_path}")
log(f"   Header lines: {sum(1 for _ in open(header_path))}")
log("")

# ------------------ Summary ------------------
summary = {
    "dataset": {
        "samples": int(len(df)),
        "features": 7,
        "classes": int(NUM_CLASSES)
    },
    "train_split": {"train": int(len(X_train)), "test": int(len(X_test))},
    "keras_nn": {
        "architecture": "Dense(32,relu) -> Dense(16,relu) -> Dense(22,softmax)",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "training_time_seconds": round(train_time, 2),
        "accuracy":  round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec), 4),
        "f1_score":  round(float(f1), 4)
    },
    "tflite_float32": {
        "size_bytes": int(fp_size),
        "size_kb": round(fp_size/1024, 2),
        "accuracy": round(float(fp_acc), 4),
        "mean_latency_ms": round(float(fp_mean_ms), 4),
        "median_latency_ms": round(float(fp_med_ms), 4)
    },
    "tflite_int8_esp32s3": {
        "size_bytes": int(int8_size),
        "size_kb": round(int8_size/1024, 2),
        "accuracy": round(float(int8_acc), 4),
        "mean_latency_ms": round(float(int8_mean_ms), 4),
        "median_latency_ms": round(float(int8_med_ms), 4),
        "max_latency_ms": round(float(int8_max_ms), 4),
        "compression_ratio": round(float(fp_size/int8_size), 2)
    }
}

with open(os.path.join(OUT_DIR, 'pipeline_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

log("="*70)
log("PIPELINE COMPLETE — All outputs saved to: " + OUT_DIR)
log("="*70)
log(json.dumps(summary, indent=2))

LOG_FILE.close()
