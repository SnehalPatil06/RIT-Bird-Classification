"""
Bird Species Classification - Training Script
Rajarambapu Institute of Technology, Rajaramnagar
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from cnn_model import build_cnn_model

# ── Configuration ─────────────────────────────────────────────────
IMAGE_SIZE    = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 50
NUM_CLASSES   = 260
LEARNING_RATE = 0.001

TRAIN_DIR = 'data/train'
VAL_DIR   = 'data/valid'
TEST_DIR  = 'data/test'

# ── Data Augmentation ──────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.15,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# ── Data Generators ────────────────────────────────────────────────
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ── Build Model ────────────────────────────────────────────────────
model = build_cnn_model(num_classes=NUM_CLASSES)
model.summary()

# ── Callbacks ─────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# ── Train ──────────────────────────────────────────────────────────
print("\n🚀 Starting training...\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# ── Save Final Model ───────────────────────────────────────────────
model.save('bird_classifier_final.h5')
print("\n✅ Training complete. Model saved as bird_classifier_final.h5")

# ── Evaluate on Test Set ───────────────────────────────────────────
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

loss, accuracy = model.evaluate(test_generator)
print(f"\n📊 Test Accuracy: {accuracy * 100:.2f}%")
print(f"📊 Test Loss:     {loss:.4f}")
