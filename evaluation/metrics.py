import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history):
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation loss
    ax1.plot(history.history['loss'], label='Training Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Model Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Quaternion Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # Plot training & validation MAE
    ax2.plot(history.history['mae'], label='Training MAE', color='blue')
    ax2.plot(history.history['val_mae'], label='Validation MAE', color='red')
    ax2.set_title('Model MAE Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    # Add some analytics as text
    best_epoch = np.argmin(history.history['val_loss'])
    min_val_loss = min(history.history['val_loss'])
    final_train_loss = history.history['loss'][-1]
    
    analytics_text = (
        f'Best Validation Loss: {min_val_loss:.4f} (Epoch {best_epoch})\n'
        f'Final Training Loss: {final_train_loss:.4f}\n'
        f'Total Epochs: {len(history.history["loss"])}'
    )
    
    plt.figtext(0.02, 0.02, analytics_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    # Print additional analytics
    print("\nTraining Analytics:")
    print("-" * 50)
    print(f"Best Validation Loss: {min_val_loss:.4f} (Epoch {best_epoch})")
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Loss Improvement: {((history.history['loss'][0] - final_train_loss) / history.history['loss'][0] * 100):.2f}%")
    print(f"Training Duration: {len(history.history['loss'])} epochs")
    
    # Calculate convergence metrics
    loss_diff = np.diff(history.history['loss'])
    convergence_rate = np.mean(np.abs(loss_diff))
    print(f"Average Convergence Rate: {convergence_rate:.6f} per epoch")
    
    # Check for overfitting
    train_val_diff = np.array(history.history['loss']) - np.array(history.history['val_loss'])
    max_overfitting = np.max(np.abs(train_val_diff))
    print(f"Maximum Train-Val Difference: {max_overfitting:.4f}")
    
    return {
        'best_epoch': best_epoch,
        'min_val_loss': min_val_loss,
        'final_train_loss': final_train_loss,
        'convergence_rate': convergence_rate,
        'max_overfitting': max_overfitting
    }

# Additional function to plot quaternion components distribution
def plot_quaternion_distribution(predictions, targets):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    # Split predictions and targets into individual quaternions
    pred_q0, pred_q1 = np.split(predictions, 2, axis=-1)
    true_q0, true_q1 = np.split(targets, 2, axis=-1)
    
    # Plot distributions for each quaternion component
    for i in range(4):
        axes[i].hist(pred_q0[:, i], bins=30, alpha=0.5, label='Predicted Q0', color='blue')
        axes[i].hist(true_q0[:, i], bins=30, alpha=0.5, label='True Q0', color='red')
        axes[i].set_title(f'Component {i+1} Distribution')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

# Usage example:
def analyze_model_performance(model, history, test_data, test_targets):
    # Plot training history
    metrics = plot_training_history(history)
    
    # Make predictions on test data
    predictions = model.predict(test_data)
    
    # Plot quaternion distributions
    #plot_quaternion_distribution(predictions, test_targets)
    
    # Calculate test set metrics
    test_loss = model.evaluate(test_data, test_targets)
    print("\nTest Set Performance:")
    print("-" * 50)
    print(f"Test Loss: {test_loss[0]:.4f}")
    print(f"Test MAE: {test_loss[1]:.4f}")
    
    return metrics