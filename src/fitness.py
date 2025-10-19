# src/fitness.py
import tensorflow as tf
from src.model import build_unet, dice_loss, dice_coef
from src.data_loader import load_data

# Load data once to be used by all fitness evaluations
(X_train, y_train), (X_val, y_val), _ = load_data(sample_size=500) # Use a small sample

class FitnessEvaluator:
    def __init__(self):
        self.param_keys = ['learning_rate', 'dropout_rate', 'num_filters']

    def evaluate(self, particle_position):
        """Trains a U-Net and returns its validation Dice Score."""
        # Decode particle position into hyperparameters
        params = dict(zip(self.param_keys, particle_position))
        params['num_filters'] = int(params['num_filters']) # Must be an integer

        # Build and compile the model
        model = build_unet((128, 128, 1), params['num_filters'], params['dropout_rate'])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                      loss=dice_loss, metrics=[dice_coef])
        
        # --- THIS IS THE BOTTLENECK ---
        # Train for a small number of epochs
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=3, batch_size=16, verbose=0)
        
        # Return the best validation dice score
        val_dice = max(history.history['val_dice_coef'])
        print(f"Params: {params} -> Val Dice: {val_dice:.4f}")
        return val_dice