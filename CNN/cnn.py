# -*- coding: utf-8 -*-
"""
CNN for Brain Tumor Object Detection with Regularization Experiments
(Correcting Optimizer Reuse and Attempting Eager Execution)

Okay, my goal here is to build a Convolutional Neural Network (CNN) using TensorFlow/Keras
to detect brain tumors in MR images, based on the provided dataset structure.

The key steps are:
1.  Set up the environment, including an output folder for results.
2.  Understand the dataset structure (assuming YOLO format for labels).
3.  Implement data loading and preprocessing using tf.data.Dataset (Native TF Ops).
4.  Define a function to build the CNN model architecture (two outputs: bbox, class).
5.  Implement different regularization techniques.
6.  Define loss functions and ensure a fresh optimizer instance is used for each model.
7.  Train the model multiple times for comparison (using run_eagerly=True for debugging).
8.  Visualize training history.
9.  Visualize prediction results.
10. Save all plots and results in the 'CNN_Outputs' folder.
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import cv2 # No longer needed for basic loading with tf.image
# import yaml # If we needed to parse the yaml file
import traceback # For detailed error printing

print("TensorFlow Version:", tf.__version__)

# --- 1. Setup Environment ---
print("--- Setting up Environment ---")
OUTPUT_DIR = 'CNN_Outputs'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: '{OUTPUT_DIR}' to save plots and results.")
else:
    print(f"Directory '{OUTPUT_DIR}' already exists.")

# Define dataset paths (focusing on axial plane for this example)
# IMPORTANT: Adjust this base path to where you extracted the dataset
DATASET_BASE_DIR = '.' # Assuming dataset folders are in the same directory as the script
# Selecting 'axial_t1wce_2_class' based on the provided file structure
PLANE_DIR = 'axial_t1wce_2_class'

# Define paths for training images and labels
TRAIN_IMAGE_DIR = os.path.join(DATASET_BASE_DIR, PLANE_DIR, 'images', 'train')
TRAIN_LABEL_DIR = os.path.join(DATASET_BASE_DIR, PLANE_DIR, 'labels', 'train')
# Define paths for validation (using the 'test' split) images and labels
VAL_IMAGE_DIR = os.path.join(DATASET_BASE_DIR, PLANE_DIR, 'images', 'test')
VAL_LABEL_DIR = os.path.join(DATASET_BASE_DIR, PLANE_DIR, 'labels', 'test')

# Check if directories exist
if not os.path.isdir(TRAIN_IMAGE_DIR) or not os.path.isdir(TRAIN_LABEL_DIR):
     print(f"ERROR: Training directories not found at {TRAIN_IMAGE_DIR} / {TRAIN_LABEL_DIR}")
     print("Please ensure the dataset is extracted and the DATASET_BASE_DIR is correct.")
     exit()
if not os.path.isdir(VAL_IMAGE_DIR) or not os.path.isdir(VAL_LABEL_DIR):
     print(f"ERROR: Validation ('test') directories not found at {VAL_IMAGE_DIR} / {VAL_LABEL_DIR}")
     print("Please ensure the dataset is extracted and the DATASET_BASE_DIR is correct.")
     exit()


# --- Configuration ---
IMG_WIDTH = 128 # Resize images to this width
IMG_HEIGHT = 128 # Resize images to this height
IMG_CHANNELS = 3 # Assuming JPG images are RGB
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
NUM_CLASSES = 1 # Just detecting 'tumor' presence/location
BATCH_SIZE = 32
EPOCHS = 30 # Set a moderate number of epochs for demonstration
LEARNING_RATE = 0.001


# --- 2. Data Loading and Preprocessing (Using Native TF Ops)---
print("\n--- Preparing Data Loaders (Using Native TF Ops) ---")

# Using the robust V3 parser
def parse_yolo_label_tf(label_path):
    """Reads the *first valid formatted line* of a YOLO format label file using TF operations."""
    label_content = tf.io.read_file(label_path)
    content_cleaned = tf.strings.regex_replace(label_content, "\r\n", "\n")
    lines = tf.strings.split(content_cleaned, sep='\n')
    lines = tf.strings.strip(lines)
    lines = tf.boolean_mask(lines, tf.strings.length(lines) > 0)
    has_lines = tf.shape(lines)[0] > 0

    def process_first_line():
        first_line = lines[0]
        line = tf.strings.regex_replace(first_line, r"\s+", " ")
        line = tf.strings.strip(line)
        parts = tf.strings.split(line, sep=' ')
        num_parts = tf.shape(parts)[0]
        has_correct_parts = tf.equal(num_parts, 5)

        def parse_correct_line():
            class_id_str = parts[0]
            bbox_coords_str = parts[1:5]
            bbox_coords = tf.strings.to_number(bbox_coords_str, out_type=tf.float32)
            class_id = tf.strings.to_number(class_id_str, out_type=tf.int32)
            is_nan_mask = tf.math.is_nan(bbox_coords)
            is_any_nan = tf.reduce_any(is_nan_mask)
            is_valid_class = tf.equal(class_id, 0)
            def return_valid_parse():
                 return bbox_coords, tf.constant([1.0], dtype=tf.float32)
            def return_zeros_nan_or_bad_class():
                 return tf.zeros((4,), dtype=tf.float32), tf.constant([0.0], dtype=tf.float32)
            return tf.cond(tf.logical_and(tf.logical_not(is_any_nan), is_valid_class),
                           return_valid_parse,
                           return_zeros_nan_or_bad_class)

        def return_zeros_invalid_format():
             return tf.zeros((4,), dtype=tf.float32), tf.constant([0.0], dtype=tf.float32)
        return tf.cond(has_correct_parts, parse_correct_line, return_zeros_invalid_format)

    def return_zeros_empty_file():
        return tf.zeros((4,), dtype=tf.float32), tf.constant([0.0], dtype=tf.float32)

    bbox_coords, confidence = tf.cond(has_lines, process_first_line, return_zeros_empty_file)
    bbox_coords.set_shape((4,))
    confidence.set_shape((1,))
    return {'bbox_output': bbox_coords, 'class_output': confidence}

def load_and_preprocess_image_tf(image_path):
    """Loads and preprocesses a single image using TF operations."""
    img_raw = tf.io.read_file(image_path)
    def decode_img():
        try:
             img_decoded = tf.image.decode_image(img_raw, channels=IMG_CHANNELS, expand_animations=False)
             img_decoded.set_shape([None, None, IMG_CHANNELS])
             return img_decoded
        except tf.errors.InvalidArgumentError:
             tf.print("Warning: Could not decode image", image_path, ". Returning zeros.")
             return tf.zeros((1, 1, IMG_CHANNELS), dtype=tf.uint8)
    def empty_img():
        return tf.zeros((1, 1, IMG_CHANNELS), dtype=tf.uint8)
    img = tf.cond(tf.strings.length(img_raw) > 0, decode_img, empty_img)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape(INPUT_SHAPE)
    return img

# Data Augmentation Layer
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomBrightness(0.2),
  layers.RandomContrast(0.2),
], name="data_augmentation")

# Create tf.data Datasets function
def create_dataset(image_dir, label_dir, batch_size, augment=False):
    """Creates a tf.data.Dataset for object detection using TF ops."""
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    label_paths = sorted(glob.glob(os.path.join(label_dir, '*.txt')))
    img_basenames = {os.path.splitext(os.path.basename(p))[0] for p in image_paths}
    lbl_basenames = {os.path.splitext(os.path.basename(p))[0] for p in label_paths}
    valid_basenames = list(img_basenames.intersection(lbl_basenames))
    if not valid_basenames: raise ValueError(f"No matching pairs in {image_dir}/{label_dir}")
    original_img_count = len(img_basenames); original_lbl_count = len(lbl_basenames)
    if len(valid_basenames) < original_img_count or len(valid_basenames) < original_lbl_count:
        print(f"Warning: Mismatch images({original_img_count}) vs labels({original_lbl_count}) in {image_dir}/{label_dir}. Using {len(valid_basenames)} matching pairs.")
    image_paths = sorted([os.path.join(image_dir, f"{name}.jpg") for name in valid_basenames])
    label_paths = sorted([os.path.join(label_dir, f"{name}.txt") for name in valid_basenames])
    if not image_paths: raise ValueError(f"No images found in {image_dir} after matching.")

    img_path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    lbl_path_ds = tf.data.Dataset.from_tensor_slices(label_paths)
    img_ds = img_path_ds.map(load_and_preprocess_image_tf, num_parallel_calls=tf.data.AUTOTUNE)
    lbl_ds = lbl_path_ds.map(parse_yolo_label_tf, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = tf.data.Dataset.zip((img_ds, lbl_ds))

    if augment:
        print("Data augmentation enabled for this dataset.")
        dataset = dataset.map(lambda img, lbl: (data_augmentation(img, training=True), lbl), num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
         dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
         print(f"Shuffling training dataset with buffer size {len(image_paths)}")

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    print(f"Created dataset from {image_dir}. Augment: {augment}. Batch size: {batch_size}")
    return dataset

# Create the datasets
try:
    train_ds_augmented = create_dataset(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, BATCH_SIZE, augment=True)
    train_ds_no_augment = create_dataset(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, BATCH_SIZE, augment=False)
    val_ds = create_dataset(VAL_IMAGE_DIR, VAL_LABEL_DIR, BATCH_SIZE, augment=False)
except ValueError as e:
     print(f"Error creating datasets: {e}")
     exit()
except Exception as e:
     print(f"Unexpected error creating datasets: {e}")
     traceback.print_exc()
     exit()

# Inspect a batch
print("\nInspecting one batch from the training dataset (with augmentation)...")
try:
    for images, labels in train_ds_augmented.take(1):
        print("Images batch shape:", images.shape)
        print("Labels batch structure:", {k: v.shape for k, v in labels.items()})
except Exception as e:
    print(f"Could not inspect dataset batch. Error: {e}")

# --- 3. Model Building ---
print("\n--- Defining Model Builder Function ---")
def build_cnn_object_detector(input_shape, num_classes,
                              regularization_type=None,
                              reg_param=None):
    """Builds the CNN model with optional regularization."""
    print(f"Building model with regularization: {regularization_type}, param: {reg_param}")
    l2_reg = None
    if regularization_type == 'l2' and reg_param is not None:
        l2_reg = tf.keras.regularizers.l2(reg_param)
        print(f"  Applying L2 regularization with lambda={reg_param}")
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    # Backbone
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2_reg)(inputs)
    if regularization_type == 'batchnorm': x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    if regularization_type == 'dropout' and reg_param is not None: x = layers.Dropout(reg_param)(x)
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2_reg)(x)
    if regularization_type == 'batchnorm': x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    if regularization_type == 'dropout' and reg_param is not None: x = layers.Dropout(reg_param)(x)
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2_reg)(x)
    if regularization_type == 'batchnorm': x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    if regularization_type == 'dropout' and reg_param is not None: x = layers.Dropout(reg_param)(x)
    # Head
    x = layers.Flatten()(x)
    x = layers.Dense(128, kernel_regularizer=l2_reg)(x)
    if regularization_type == 'batchnorm': x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if regularization_type == 'dropout' and reg_param is not None: x = layers.Dropout(max(0.2, reg_param))(x)
    # Outputs
    bbox_output = layers.Dense(4, activation='sigmoid', name='bbox_output')(x)
    class_output = layers.Dense(num_classes, activation='sigmoid', name='class_output')(x)
    # Model
    model_name = f"cnn_detector_{regularization_type or 'baseline'}"
    if reg_param is not None: model_name += f"_{str(reg_param).replace('.','')}"
    model = tf.keras.Model(inputs=inputs, outputs=[bbox_output, class_output], name=model_name)
    print("Model built successfully.")
    model.summary()
    return model

# --- 4. Loss Functions ---
print("\n--- Defining Loss Functions ---")
bbox_loss_fn = tf.keras.losses.MeanSquaredError(name="bbox_mse_loss")
class_loss_fn = tf.keras.losses.BinaryCrossentropy(name="class_bce_loss")
loss_weights_dict = {'bbox_output': 1.0, 'class_output': 1.0}

# NOTE: Optimizer is now created inside train_and_evaluate

# --- 5. Training and Evaluation Loop ---
print("\n--- Starting Training & Evaluation Experiments ---")
results = {}

def train_and_evaluate(model_name, model, train_dataset, val_dataset, epochs):
    """Compiles, trains, and evaluates a model."""
    print(f"\n--- Experiment: {model_name} ---")

    # <<< CORRECTION: Create a NEW optimizer instance for each model >>>
    optimizer_local = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # <<< END CORRECTION >>>

    # Compile the model using the new optimizer instance
    model.compile(optimizer=optimizer_local, # Use the local optimizer
                  loss={'bbox_output': bbox_loss_fn, 'class_output': class_loss_fn},
                  loss_weights=loss_weights_dict,
                  metrics={'class_output': 'accuracy'},
                  run_eagerly=True) # Keep eager for now for debugging

    print(f"Training model for {epochs} epochs (running eagerly)...") # Note about eager run
    try:
        train_size_tensor = train_dataset.cardinality()
        val_size_tensor = val_dataset.cardinality()
        train_size = train_size_tensor.numpy() if train_size_tensor != tf.data.UNKNOWN_CARDINALITY else 'unknown'
        val_size = val_size_tensor.numpy() if val_size_tensor != tf.data.UNKNOWN_CARDINALITY else 'unknown'
        print(f"Train dataset size (batches): {train_size}, Val dataset size (batches): {val_size}")

        effective_val_ds = val_dataset
        if val_size == 0 or val_size == 'unknown' or (isinstance(val_size, int) and val_size < 1):
             print(f"Warning: Validation dataset for {model_name} is empty or invalid ({val_size}). Proceeding without validation.")
             effective_val_ds = None

        history = model.fit(
            train_dataset,
            validation_data=effective_val_ds,
            epochs=epochs,
            verbose=1,
        )
        results[model_name] = {'history': history, 'model': model}
        print(f"Finished training for {model_name}")
    except Exception as e:
        print(f"ERROR during training for {model_name}: {e}")
        traceback.print_exc()
        results[model_name] = {'history': None, 'model': model, 'error': e}

# --- Running Experiments ---
# Experiment 1: Baseline (No Augmentation, No Explicit Regularization)
model_baseline_noaug = build_cnn_object_detector(INPUT_SHAPE, NUM_CLASSES, regularization_type=None)
train_and_evaluate("Baseline (No Aug)", model_baseline_noaug, train_ds_no_augment, val_ds, EPOCHS)

# Experiment 2: Baseline + Data Augmentation
model_baseline_aug = build_cnn_object_detector(INPUT_SHAPE, NUM_CLASSES, regularization_type=None)
train_and_evaluate("Baseline (With Aug)", model_baseline_aug, train_ds_augmented, val_ds, EPOCHS)

# Experiment 3: L2 Regularization (with Augmentation)
l2_lambda = 0.001
model_l2 = build_cnn_object_detector(INPUT_SHAPE, NUM_CLASSES, regularization_type='l2', reg_param=l2_lambda)
train_and_evaluate(f"L2 Reg (lambda={l2_lambda})", model_l2, train_ds_augmented, val_ds, EPOCHS)

# Experiment 4: Dropout (with Augmentation)
dropout_rate = 0.3
model_dropout = build_cnn_object_detector(INPUT_SHAPE, NUM_CLASSES, regularization_type='dropout', reg_param=dropout_rate)
train_and_evaluate(f"Dropout (rate={dropout_rate})", model_dropout, train_ds_augmented, val_ds, EPOCHS)

# Experiment 5: Batch Normalization (with Augmentation)
model_bn = build_cnn_object_detector(INPUT_SHAPE, NUM_CLASSES, regularization_type='batchnorm')
train_and_evaluate("Batch Norm", model_bn, train_ds_augmented, val_ds, EPOCHS)


# --- 6. Visualize Training History ---
print("\n--- Visualizing Training Histories ---")
if not results:
    print("No models were successfully trained. Skipping history visualization.")
else:
    plt.figure(figsize=(15, 10))
    # Plot Total Loss
    plt.subplot(2, 2, 1)
    for name, result in results.items():
        if result.get('history'):
            plt.plot(result['history'].history['loss'], label=f'{name} Train Loss')
            if 'val_loss' in result['history'].history:
                 plt.plot(result['history'].history['val_loss'], label=f'{name} Val Loss', linestyle='--')
    plt.title('Total Model Loss Comparison'); plt.ylabel('Loss'); plt.xlabel('Epoch'); plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels();
    if handles: plt.legend(handles, labels, fontsize='small')

    # Plot Bbox Loss
    plt.subplot(2, 2, 2)
    for name, result in results.items():
        if result.get('history'):
            bbox_loss_key = next((k for k in result['history'].history if 'bbox_output_loss' in k), None)
            val_bbox_loss_key = next((k for k in result['history'].history if 'val_bbox_output_loss' in k), None)
            if bbox_loss_key: plt.plot(result['history'].history[bbox_loss_key], label=f'{name} Train BBox Loss')
            if val_bbox_loss_key: plt.plot(result['history'].history[val_bbox_loss_key], label=f'{name} Val BBox Loss', linestyle='--')
    plt.title('Bounding Box Loss Comparison'); plt.ylabel('BBox Loss (MSE)'); plt.xlabel('Epoch'); plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels();
    if handles: plt.legend(handles, labels, fontsize='small')

    # Plot Class Loss
    plt.subplot(2, 2, 3)
    for name, result in results.items():
        if result.get('history'):
            class_loss_key = next((k for k in result['history'].history if 'class_output_loss' in k), None)
            val_class_loss_key = next((k for k in result['history'].history if 'val_class_output_loss' in k), None)
            if class_loss_key: plt.plot(result['history'].history[class_loss_key], label=f'{name} Train Class Loss')
            if val_class_loss_key: plt.plot(result['history'].history[val_class_loss_key], label=f'{name} Val Class Loss', linestyle='--')
    plt.title('Classification Loss Comparison'); plt.ylabel('Class Loss (Binary CE)'); plt.xlabel('Epoch'); plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels();
    if handles: plt.legend(handles, labels, fontsize='small')

    # Plot Class Accuracy
    plt.subplot(2, 2, 4)
    for name, result in results.items():
        if result.get('history'):
            acc_key = next((k for k in result['history'].history if 'class_output_accuracy' in k), None)
            val_acc_key = next((k for k in result['history'].history if 'val_class_output_accuracy' in k), None)
            if acc_key: plt.plot(result['history'].history[acc_key], label=f'{name} Train Class Acc')
            if val_acc_key: plt.plot(result['history'].history[val_acc_key], label=f'{name} Val Class Acc', linestyle='--')
    plt.title('Classification Accuracy Comparison'); plt.ylabel('Accuracy'); plt.xlabel('Epoch'); plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels();
    if handles: plt.legend(handles, labels, fontsize='small')

    plt.tight_layout()
    history_plot_path = os.path.join(OUTPUT_DIR, '01_training_history_comparison.png')
    plt.savefig(history_plot_path)
    plt.close()
    print(f"Saved training history comparison plot to: {history_plot_path}")

# --- 7. Visualize Predictions ---
print("\n--- Visualizing Predictions on Validation Set ---")
def plot_predictions(model, dataset, model_name, num_images=5):
    """Plots predictions vs ground truth for a few images."""
    plt.figure(figsize=(15, 5 * num_images))
    plot_index = 1
    try:
        if not isinstance(dataset, tf.data.Dataset):
             print(f"Error: Expected dataset for {model_name}, got {type(dataset)}")
             return
        val_size_tensor = dataset.cardinality()
        val_size = val_size_tensor.numpy() if val_size_tensor != tf.data.UNKNOWN_CARDINALITY else -1 # Use -1 for unknown
        if val_size == 0:
             print(f"Validation dataset for {model_name} is empty. Cannot plot predictions.")
             return

        iterator = iter(dataset)
        images, labels = next(iterator)

        predictions = model.predict(images)
        pred_bboxes = predictions[0]
        pred_confidences = predictions[1]
        gt_bboxes = labels['bbox_output'].numpy()
        gt_confidences = labels['class_output'].numpy()

        for i in range(min(num_images, images.shape[0])):
            img = images[i].numpy()
            ax = plt.subplot(num_images, 1, plot_index)
            plt.imshow(img)
            plt.title(f"{model_name} - Image {i}")
            plt.axis('off')
            img_h, img_w = img.shape[0], img.shape[1]

            # Draw Ground Truth Box
            if gt_confidences[i] > 0.5:
                xc, yc, w, h = gt_bboxes[i]
                xmin = (xc - w / 2) * img_w; ymin = (yc - h / 2) * img_h
                box_w = w * img_w; box_h = h * img_h
                xmin = max(0, xmin); ymin = max(0, ymin)
                box_w = min(img_w - xmin - 1, box_w); box_h = min(img_h - ymin - 1, box_h)
                if box_w > 0 and box_h > 0:
                    gt_rect = patches.Rectangle((xmin, ymin), box_w, box_h, linewidth=2, edgecolor='g', facecolor='none')
                    ax.add_patch(gt_rect)
                    ax.text(xmin, ymin - 5, 'Tumor (GT)', color='green', fontsize=9, bbox=dict(facecolor='white', alpha=0.5, pad=0.1))

            # Draw Predicted Box
            pred_conf = pred_confidences[i][0]
            if pred_conf > 0.5:
                xc, yc, w, h = pred_bboxes[i]
                xmin = (xc - w / 2) * img_w; ymin = (yc - h / 2) * img_h
                box_w = w * img_w; box_h = h * img_h
                xmin = max(0, xmin); ymin = max(0, ymin)
                box_w = min(img_w - xmin - 1, box_w); box_h = min(img_h - ymin - 1, box_h)
                if box_w > 0 and box_h > 0:
                    pred_rect = patches.Rectangle((xmin, ymin), box_w, box_h, linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(pred_rect)
                    ax.text(xmin, ymin + box_h + 5, f'Tumor ({pred_conf:.2f})', color='red', fontsize=9, bbox=dict(facecolor='white', alpha=0.5, pad=0.1))
            plot_index += 1

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        safe_model_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('.', '').replace('/','')
        pred_plot_path = os.path.join(OUTPUT_DIR, f"02_predictions_{safe_model_name}.png")
        plt.savefig(pred_plot_path)
        plt.close()
        print(f"Saved prediction visualization plot to: {pred_plot_path}")

    except StopIteration:
        print(f"Could not get a batch from the dataset for {model_name} to plot predictions (Dataset might be empty or exhausted).")
    except Exception as e:
        print(f"Error during prediction visualization for {model_name}: {e}")
        traceback.print_exc()

# Visualize predictions for each model that trained successfully
for name, result in results.items():
    if result.get('history') and result.get('model'):
        # Recreate val_ds for visualization if needed, handle potential errors
        try:
            val_ds_vis = create_dataset(VAL_IMAGE_DIR, VAL_LABEL_DIR, BATCH_SIZE, augment=False)
            plot_predictions(result['model'], val_ds_vis, name, num_images=4)
        except Exception as e:
             print(f"Could not visualize predictions for {name} due to dataset error: {e}")

# --- Final Comparison Guide ---
print("\n--- All Experiments Finished ---")
print(f"Check the '{OUTPUT_DIR}' folder for all generated plots.")
print("\nComparison Guide:")
print("1. Look at '01_training_history_comparison.png':")
print("   - Compare validation loss curves (dashed lines). Lower is generally better.")
print("   - Does any regularization method significantly reduce the gap between training and validation loss (overfitting)?")
print("   - Compare validation accuracy curves (if applicable). Higher is better.")
print("   - Does Batch Norm lead to faster convergence (loss drops quicker)?")
print("2. Look at the '02_predictions_*.png' files:")
print("   - Visually compare the predicted red boxes to the ground truth green boxes.")
print("   - Does regularization (L2, Dropout) or Batch Norm lead to more accurate box placement or confidence scores?")
print("   - How does the 'Baseline (No Aug)' compare to 'Baseline (With Aug)'? Does augmentation improve robustness?")