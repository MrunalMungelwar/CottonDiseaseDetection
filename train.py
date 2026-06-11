"""
CNN Training Script for Cotton Plant Disease Detection.

Custom CNN architecture (no transfer learning).
Includes data augmentation, train-validation-test split, metrics display,
confusion matrix generation, class balancing, and model saving.
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DATASET_DIR,
    EPOCHS,
    IMG_CHANNELS,
    IMG_HEIGHT,
    IMG_WIDTH,
    LEARNING_RATE,
    MODEL_PATH,
    RANDOM_SEED,
    TRAINING_PLOTS_DIR,
    VALIDATION_SPLIT,
    TEST_SPLIT,
)
from dataset_utils import get_all_image_paths, get_dataset_summary, validate_dataset_available

# Reproducibility
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def build_cnn_model(num_classes: int = 4) -> tf.keras.Model:
    """
    Build a custom CNN architecture for cotton disease classification.

    Architecture:
        - 5 Convolution blocks with BatchNorm, ReLU, MaxPooling, Dropout, and L2 Reg
        - GlobalAveragePooling2D
        - Dense layers with Dropout
        - Softmax output

    Args:
        num_classes: Number of output classes.

    Returns:
        Compiled Keras model.
    """
    model = models.Sequential(name="CottonDiseaseCNN")
    l2_reg = regularizers.l2(0.001)

    # Block 1: 32 filters
    model.add(layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))
    model.add(layers.Conv2D(32, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(32, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    # Block 2: 64 filters
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    # Block 3: 128 filters
    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 4: 256 filters
    model.add(layers.Conv2D(256, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(256, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 5: 512 filters
    model.add(layers.Conv2D(512, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(512, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # Classification head
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation="relu", kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def clean_corrupted_images(file_paths, labels):
    """
    Remove corrupted images to prevent training errors.
    Returns cleaned paths and labels.
    """
    clean_paths = []
    clean_labels = []
    print("Checking for corrupted images...")
    for path, label in zip(file_paths, labels):
        try:
            # Check if image can be opened by cv2 (catches corrupted/empty files)
            img = cv2.imread(path)
            if img is not None:
                clean_paths.append(path)
                clean_labels.append(label)
            else:
                print(f"Deleting corrupted image (unreadable): {path}")
                os.remove(path)
        except Exception as e:
            print(f"Error reading {path}, deleting: {e}")
            try:
                os.remove(path)
            except:
                pass
    return clean_paths, clean_labels


def _load_images_from_paths(file_paths, labels):
    """Load and preprocess images from file paths into numpy arrays."""
    images = []
    valid_labels = []
    for path, label in zip(file_paths, labels):
        try:
            img = tf.keras.utils.load_img(
                path, target_size=(IMG_HEIGHT, IMG_WIDTH)
            )
            img_array = tf.keras.utils.img_to_array(img)
            images.append(img_array)
            valid_labels.append(label)
        except Exception as e:
            print(f"Warning: Skipping {path}: {e}")
    return np.array(images), np.array(valid_labels)


def create_data_generators():
    """
    Create training, validation, and test data generators.
    Also computes and returns class weights.
    """
    file_paths, labels = get_all_image_paths()
    file_paths, labels = clean_corrupted_images(file_paths, labels)

    # Stratified Splits: Train, Val, Test
    # First, split into train and temp (val+test)
    val_test_ratio = VALIDATION_SPLIT + TEST_SPLIT
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths, labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=RANDOM_SEED
    )

    # Now split temp into val and test
    test_ratio_relative = TEST_SPLIT / val_test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=test_ratio_relative,
        stratify=temp_labels,
        random_state=RANDOM_SEED
    )

    print("Loading training images...")
    x_train, y_train = _load_images_from_paths(train_paths, train_labels)
    print("Loading validation images...")
    x_val, y_val = _load_images_from_paths(val_paths, val_labels)
    print("Loading test images...")
    x_test, y_test = _load_images_from_paths(test_paths, test_labels)

    # Convert labels to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(CLASS_NAMES))
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=len(CLASS_NAMES))
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=len(CLASS_NAMES))

    # Compute Class Weights to handle imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights_dict = {cls: weight for cls, weight in zip(classes, weights)}
    print(f"Computed Class Weights: {class_weights_dict}")

    # Training data augmentation (Relaxed)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode="nearest",
    )

    # Validation and Test: only rescaling
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow(
        x_train, y_train_cat,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED,
    )

    validation_generator = val_test_datagen.flow(
        x_val, y_val_cat,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_generator = val_test_datagen.flow(
        x_test, y_test_cat,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Attach metadata for logging
    train_generator.samples = len(x_train)
    validation_generator.samples = len(x_val)
    test_generator.samples = len(x_test)
    train_generator.class_indices = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    return train_generator, validation_generator, test_generator, class_weights_dict


def plot_training_history(history) -> None:
    """Plot and save training/validation accuracy and loss curves."""
    os.makedirs(TRAINING_PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(history.history["accuracy"], label="Training Accuracy", color="#2E7D32")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy", color="#FF6F00")
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(history.history["loss"], label="Training Loss", color="#2E7D32")
    axes[1].plot(history.history["val_loss"], label="Validation Loss", color="#FF6F00")
    axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(TRAINING_PLOTS_DIR, "training_history.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training history plot saved to: {plot_path}")


def evaluate_and_plot_confusion_matrix(model, test_generator) -> None:
    """Evaluate model on Test set, generate metrics, and save confusion matrix."""
    os.makedirs(TRAINING_PLOTS_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    # Reset generator and get predictions
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_generator.y, axis=1) if len(test_generator.y.shape) > 1 else test_generator.y

    cm = confusion_matrix(true_classes, predicted_classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix (Test Set)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    fig.tight_layout()
    cm_path = os.path.join(TRAINING_PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

    # Print classification report
    print("\nDetailed Test Classification Report (Precision, Recall, F1-Score):")
    report = classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES)
    print(report)


def validate_dataset() -> bool:
    """Check that dataset images are available for all classes."""
    if not os.path.exists(DATASET_DIR):
        print(f"ERROR: Dataset directory not found: {DATASET_DIR}")
        return False

    is_valid, message = validate_dataset_available()
    summary = get_dataset_summary()

    print("Dataset summary:")
    for cls, count in summary.items():
        print(f"  {cls}: {count} images")
    print(message)

    return is_valid


def save_untrained_model() -> None:
    """Save an untrained model with correct architecture for app bootstrap."""
    model = build_cnn_model(num_classes=len(CLASS_NAMES))
    model.save(MODEL_PATH)
    print(f"Untrained model architecture saved to: {MODEL_PATH}")
    print("Note: Train the model with your dataset for accurate predictions.")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Cotton Plant Disease Detection - CNN Training (Enhanced Pipeline)")
    print("=" * 60)

    if not validate_dataset():
        print("\nSaving untrained model architecture for app compatibility...")
        save_untrained_model()
        sys.exit(1)

    # Build model
    print("\nBuilding custom CNN model...")
    model = build_cnn_model(num_classes=len(CLASS_NAMES))
    model.summary()

    # Create data generators
    print("\nPreparing dataset splits and augmentation...")
    train_gen, val_gen, test_gen, class_weights = create_data_generators()
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Classes: {train_gen.class_indices}")

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # Train
    print(f"\nTraining for up to {EPOCHS} epochs...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Display final metrics
    print("\n" + "=" * 60)
    print("TRAINING VALIDATION RESULTS")
    print("=" * 60)
    print(f"Final Training Accuracy:   {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final Training Loss:       {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss:     {history.history['val_loss'][-1]:.4f}")

    # Save final model
    # (Note: Best weights are already restored due to EarlyStopping restore_best_weights=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # Generate plots & Evaluate on Test set
    plot_training_history(history)
    evaluate_and_plot_confusion_matrix(model, test_gen)

    print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()
