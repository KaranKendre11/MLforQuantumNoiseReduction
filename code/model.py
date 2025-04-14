import tensorflow as tf
from tensorflow.keras import layers, models
import scipy.linalg
import matplotlib.pyplot as plt

def build_cnn_model(input_shape):
    """
    Build a CNN model for quantum error correction.
    
    Args:
        input_shape: Shape of input data (height, width, channels)
    
    Returns:
        A compiled Keras model
    """
    model = models.Sequential([
        tf.keras.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 4 (Decoder begins)
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 5
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 6 
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)), 
        layers.Dropout(0.25),

        # Output
        layers.Conv2D(2, (3, 3), activation='linear', padding='same')
    ])
    
    model.compile(optimizer='adam', loss=combined_loss, metrics=['mae'])
    return model

def approximate_fidelity_loss(y_true, y_pred):
    real_true = y_true[..., 0]
    imag_true = y_true[..., 1]
    real_pred = y_pred[..., 0]
    imag_pred = y_pred[..., 1]

    # Reconstruct complex tensors
    rho_true = tf.complex(real_true, imag_true)
    rho_pred = tf.complex(real_pred, imag_pred)

    # Flatten for batchwise inner product
    rho_true_flat = tf.reshape(rho_true, (tf.shape(rho_true)[0], -1))
    rho_pred_flat = tf.reshape(rho_pred, (tf.shape(rho_pred)[0], -1))

    # Frobenius inner product and norm-based approximation
    inner_prod = tf.reduce_sum(tf.math.conj(rho_true_flat) * rho_pred_flat, axis=1)
    norm_true = tf.reduce_sum(tf.math.conj(rho_true_flat) * rho_true_flat, axis=1)
    norm_pred = tf.reduce_sum(tf.math.conj(rho_pred_flat) * rho_pred_flat, axis=1)

    # Fidelity = |<ρ_true, ρ_pred>|^2 / (||ρ_true|| * ||ρ_pred||)
    overlap = tf.math.real(inner_prod) / (tf.sqrt(tf.math.real(norm_true) * tf.math.real(norm_pred)) + 1e-8)

    return 1.0 - tf.reduce_mean(overlap)

def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    fid_loss = approximate_fidelity_loss(y_true, y_pred)
    return mse + 0.5 * fid_loss

def plot_training_history(history):
    """
    Plot the training history of the model.
    
    Args:
        history: The history object returned by model.fit()
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()