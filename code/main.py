import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import scipy.linalg

from data_generation import (
    prepare_dataset, preprocess_for_cnn, calculate_fidelity
)
from model import build_cnn_model, plot_training_history

def visualize_density_matrices(original_dm, noisy_dm, corrected_dm, index=0):
    """
    Visualize original, noisy, and corrected density matrices.
    
    Args:
        original_dm: Original (clean) density matrix
        noisy_dm: Noisy density matrix
        corrected_dm: CNN-corrected density matrix
        index: Index of the sample to visualize
    """

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Extract matrix components for selected sample
    real_orig = np.real(original_dm[index])
    imag_orig = np.imag(original_dm[index])
    real_noisy = np.real(noisy_dm[index])
    imag_noisy = np.imag(noisy_dm[index])
    real_corr = corrected_dm[index, :, :, 0]
    imag_corr = corrected_dm[index, :, :, 1]

    # Dynamic color scaling
    vmin_real = np.min([real_orig.min(), real_noisy.min(), real_corr.min()])
    vmax_real = np.max([real_orig.max(), real_noisy.max(), real_corr.max()])
    vmin_imag = np.min([imag_orig.min(), imag_noisy.min(), imag_corr.min()])
    vmax_imag = np.max([imag_orig.max(), imag_noisy.max(), imag_corr.max()])

    # Plot Real parts
    im1 = axes[0, 0].imshow(real_orig, cmap='RdBu', vmin=vmin_real, vmax=vmax_real)
    axes[0, 0].set_title('Original DM (Real)')
    fig.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(real_noisy, cmap='RdBu', vmin=vmin_real, vmax=vmax_real)
    axes[0, 1].set_title('Noisy DM (Real)')
    fig.colorbar(im2, ax=axes[0, 1])

    im3 = axes[0, 2].imshow(real_corr, cmap='RdBu', vmin=vmin_real, vmax=vmax_real)
    axes[0, 2].set_title('Corrected DM (Real)')
    fig.colorbar(im3, ax=axes[0, 2])

    # Plot Imaginary parts
    im4 = axes[1, 0].imshow(imag_orig, cmap='RdBu', vmin=vmin_imag, vmax=vmax_imag)
    axes[1, 0].set_title('Original DM (Imaginary)')
    fig.colorbar(im4, ax=axes[1, 0])

    im5 = axes[1, 1].imshow(imag_noisy, cmap='RdBu', vmin=vmin_imag, vmax=vmax_imag)
    axes[1, 1].set_title('Noisy DM (Imaginary)')
    fig.colorbar(im5, ax=axes[1, 1])

    im6 = axes[1, 2].imshow(imag_corr, cmap='RdBu', vmin=vmin_imag, vmax=vmax_imag)
    axes[1, 2].set_title('Corrected DM (Imaginary)')
    fig.colorbar(im6, ax=axes[1, 2])

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

def plot_fidelity_comparison(original_dms, noisy_dms, corrected_dms, noise_types, noise_levels):
    """
    Plot fidelities before and after error correction for different noise types and levels.
    
    Args:
        original_dms: Original (clean) density matrices
        noisy_dms: Noisy density matrices
        corrected_dms: CNN-corrected density matrices
        noise_types: List of noise types for each sample
        noise_levels: List of noise levels for each sample
    """
    # Calculate fidelities
    noisy_fidelities = []
    corrected_fidelities = []
    
    for i in range(len(original_dms)):
        # Convert corrected output back to complex form
        corrected_complex = corrected_dms[i, :, :, 0] + 1j * corrected_dms[i, :, :, 1]
        
        # Calculate fidelity between original and noisy
        noisy_fid = calculate_fidelity(original_dms[i], noisy_dms[i])
        
        # Calculate fidelity between original and corrected
        corrected_fid = calculate_fidelity(original_dms[i], corrected_complex)
        
        noisy_fidelities.append(noisy_fid)
        corrected_fidelities.append(corrected_fid)
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Noise Type': [t for t, _ in noise_types],
        'Noise Level': [l for _, l in noise_levels],
        'Noisy Fidelity': noisy_fidelities,
        'Corrected Fidelity': corrected_fidelities
    })
    
    # Calculate improvement column here
    df['Improvement'] = df['Corrected Fidelity'] - df['Noisy Fidelity']
    
    # Plot by noise type
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Noise Type', y='value', hue='variable',
                data=pd.melt(df, id_vars=['Noise Type'],
                             value_vars=['Noisy Fidelity', 'Corrected Fidelity']))
    plt.title('Fidelity Comparison by Noise Type')
    plt.ylabel('Fidelity')
    plt.ylim(0, 1)
    
    # Plot by noise level
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Noise Level', y='value', hue='variable',
                data=pd.melt(df, id_vars=['Noise Level'],
                             value_vars=['Noisy Fidelity', 'Corrected Fidelity']))
    plt.title('Fidelity Comparison by Noise Level')
    plt.ylabel('Fidelity')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()


    correlation_matrix = df[['Noise Level', 'Noisy Fidelity', 'Corrected Fidelity']].corr()
    print(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Between Noise Level and Fidelities')
    plt.show()


    fidelity_improvement = np.array(corrected_fidelities) - np.array(noisy_fidelities)
    plt.hist(fidelity_improvement, bins=20)
    plt.title("Fidelity Improvement Distribution")
    plt.xlabel("Corrected - Noisy Fidelity")
    plt.ylabel("Count")
    plt.show()
    
    return df

def plot_noise_classification_results(y_true, y_pred, noise_types):
    """
    Plot confusion matrix for noise classification.
    
    Args:
        y_true: True noise type labels
        y_pred: Predicted noise type labels
        noise_types: List of noise type names
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=noise_types, yticklabels=noise_types)
    plt.title('Noise Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    print(classification_report(y_true, y_pred, target_names=noise_types))

def plot_sample_table(data, title):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.axis('off')
    ax.set_title(title, fontweight='bold')
    table_plot = ax.table(cellText=data.values,
                        colLabels=data.columns,
                        loc='center',
                        cellLoc='center')
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(9)
    plt.tight_layout()
    plt.show()

def main():
    # Parameters
    num_qubits = 5  # Small enough for simulation
    num_circuits = 10000
    
    noise_types = ['depolarizing', 'amplitude_damping', 'phase_damping', 'bitflip', 'mixed']
    noise_levels = [0.05, 0.1, 0.15, 0.2]


    # Step 1: Generate dataset
    print("Generating dataset...")
    X_clean, X_noisy, y_labels = prepare_dataset(
        num_circuits=num_circuits,
        num_qubits=num_qubits,
        noise_types=noise_types,
        noise_levels=noise_levels,
        seed=42
    )
    
    # Step 2: Preprocess data for CNN
    X, y = preprocess_for_cnn(X_clean, X_noisy)
    
    # Step 3: Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_clean_train, X_clean_test = train_test_split(X_clean, test_size=0.2, random_state=42)
    X_noisy_train, X_noisy_test = train_test_split(X_noisy, test_size=0.2, random_state=42)
    y_labels_train, y_labels_test = train_test_split(y_labels, test_size=0.2, random_state=42)
    
    # Step 4: Build and train the error correction model
    print("Building and training the error correction model...")
    input_shape = X_train.shape[1:]  # (height, width, channels)
    error_correction_model = build_cnn_model(input_shape)
    
    history = error_correction_model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    # Step 5: Evaluate the model
    print("Evaluating the model...")
    test_loss, test_mae = error_correction_model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
    
    # Step 6: Make predictions
    corrected_dms = error_correction_model.predict(X_test)

    # Optional: Convert predicted real+imag back to complex for visual alignment
    corrected_dms_complex = corrected_dms[:, :, :, 0] + 1j * corrected_dms[:, :, :, 1]

    # Visualize a sample density matrix
    corrected_fidelities = []
    noisy_fidelities = []

    for i in range(len(X_clean_test)):
        true_dm = X_clean_test[i]
        noisy_dm = X_noisy_test[i]
        corr_dm = corrected_dms_complex[i]

        noisy_fidelities.append(calculate_fidelity(true_dm, noisy_dm))
        corrected_fidelities.append(calculate_fidelity(true_dm, corr_dm))

    # Visualize the sample with best fidelity improvement
    best_idx = np.argmax(np.array(corrected_fidelities) - np.array(noisy_fidelities))
    visualize_density_matrices(X_clean_test, X_noisy_test, corrected_dms, index=best_idx)
    
    # Step 7: Visualize results
    print("Visualizing results...")
    # Plot training history
    plot_training_history(history)

    df = plot_fidelity_comparison(X_clean_test, X_noisy_test, corrected_dms, y_labels_test, y_labels_test)
    
    # 1. Avg Fidelity by Noise Type
    table1 = df.groupby('Noise Type').agg({
        'Noisy Fidelity': 'mean',
        'Corrected Fidelity': 'mean',
        'Improvement': 'mean'
    }).round(3)

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axis('off')
    ax.set_title("Table 1: Avg Fidelity by Noise Type", fontweight='bold')
    table_plot = ax.table(cellText=table1.values,
                        colLabels=table1.columns,
                        rowLabels=table1.index,
                        loc='center',
                        cellLoc='center')
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(10)
    plt.tight_layout()
    plt.show()


    # 2. Avg Fidelity by Noise Level
    table2 = df.groupby('Noise Level').agg({
        'Noisy Fidelity': 'mean',
        'Corrected Fidelity': 'mean',
        'Improvement': 'mean'
    }).round(3)

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axis('off')
    ax.set_title("Table 2: Avg Fidelity by Noise Level", fontweight='bold')
    table_plot = ax.table(cellText=table2.values,
                        colLabels=table2.columns,
                        rowLabels=table2.index,
                        loc='center',
                        cellLoc='center')
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(10)
    plt.tight_layout()
    plt.show()


    # 3. Top & Bottom 5
    # Note: 'Improvement' column is already calculated in plot_fidelity_comparison
    top_5 = df.sort_values('Improvement', ascending=False).head(5).round(3)
    bottom_5 = df.sort_values('Improvement').head(5).round(3)

    plot_sample_table(top_5[['Noise Type', 'Noise Level', 'Noisy Fidelity', 'Corrected Fidelity', 'Improvement']],
                    "Table 3A: Top 5 Samples (Best Recovery)")

    plot_sample_table(bottom_5[['Noise Type', 'Noise Level', 'Noisy Fidelity', 'Corrected Fidelity', 'Improvement']],
                    "Table 3B: Bottom 5 Samples (Worst Recovery)")


if __name__ == "__main__":
    main()