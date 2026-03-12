"""
Bird Species Classification - CNN Model Architecture
Rajarambapu Institute of Technology, Rajaramnagar
Department of Computer Engineering

Team:
  - Vaibhav Raju Kolekar       (1804034)
  - Shubham Shankar Patil      (1804066)
  - Snehal Rajgonda Patil      (1954011)
  - Bhagyashri Vijay Suryawanshi (1954015)

Guide: Prof. P.R. Gavali
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(num_classes=260, input_shape=(224, 224, 3)):
    """
    Build the CNN model for bird species classification.

    Architecture:
        Input (224x224x3)
        -> Conv2D + ReLU
        -> MaxPooling2D
        -> Conv2D + ReLU
        -> MaxPooling2D
        -> Conv2D + ReLU
        -> MaxPooling2D
        -> Flatten
        -> Dense (1024) + ReLU + Dropout
        -> Dense (num_classes) + Softmax

    Args:
        num_classes (int): Number of bird species to classify. Default: 260
        input_shape (tuple): Input image shape. Default: (224, 224, 3)

    Returns:
        model: Compiled Keras model
    """

    model = models.Sequential([

        # Block 1 - Convolutional + Pooling
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=input_shape, name='conv_block1'),
        layers.MaxPooling2D((2, 2), name='pool1'),

        # Block 2 - Convolutional + Pooling
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_block2'),
        layers.MaxPooling2D((2, 2), name='pool2'),

        # Block 3 - Convolutional + Pooling
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_block3'),
        layers.MaxPooling2D((2, 2), name='pool3'),

        # Block 4 - Convolutional + Pooling
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_block4'),
        layers.MaxPooling2D((2, 2), name='pool4'),

        # Flatten
        layers.Flatten(name='flatten'),

        # Fully Connected Layers
        layers.Dense(1024, activation='relu', name='fc1'),
        layers.Dropout(0.5, name='dropout1'),
        layers.Dense(512, activation='relu', name='fc2'),
        layers.Dropout(0.3, name='dropout2'),

        # Output Layer
        layers.Dense(num_classes, activation='softmax', name='output'),
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_model_summary():
    model = build_cnn_model()
    model.summary()
    return model


if __name__ == '__main__':
    model = get_model_summary()
    print(f"\nTotal parameters: {model.count_params():,}")
