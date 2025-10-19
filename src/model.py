# src/model.py
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model

def dice_coef(y_true, y_pred, smooth=1e-6):
    """Dice Coefficient metric."""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice Loss function."""
    return 1 - dice_coef(y_true, y_pred)

def build_unet(input_shape, num_filters, dropout_rate):
    """Builds a U-Net model with configurable hyperparameters."""
    inputs = layers.Input(input_shape)
    
    # Encoder
    c1 = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Dropout(dropout_rate)(c1)
    c1 = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # ... (A full U-Net would have more layers, this is a simplified example) ...
    c2 = layers.Conv2D(num_filters * 2, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Dropout(dropout_rate)(c2)
    c2 = layers.Conv2D(num_filters * 2, (3, 3), activation='relu', padding='same')(c2)

    # Decoder
    u3 = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(c2)
    u3 = layers.concatenate([u3, c1])
    c3 = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(u3)
    c3 = layers.Dropout(dropout_rate)(c3)
    c3 = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(c3)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c3)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model