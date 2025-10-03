 
"""
Utility Functions for Crop Disease Detection Project
Helper functions for visualization, data processing, and model analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import json
from PIL import Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# IMAGE PROCESSING UTILITIES
# ============================================================================

def load_image(image_path, target_size=(224, 224), normalize=True):
    """
    Load and preprocess a single image
    
    Args:
        image_path: Path to image file
        target_size: Target dimensions (height, width)
        normalize: Whether to normalize to [0, 1]
    
    Returns:
        Preprocessed numpy array
    """
    try:
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(target_size)
        
        # Convert to array
        img_array = np.array(img)
        
        # Normalize
        if normalize:
            img_array = img_array / 255.0
        
        return img_array
    
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def save_image(image_array, save_path, denormalize=True):
    """
    Save numpy array as image
    
    Args:
        image_array: Numpy array (H, W, C)
        save_path: Path to save image
        denormalize: If True, assumes array is in [0, 1] range
    """
    if denormalize and image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    
    img = Image.fromarray(image_array)
    img.save(save_path)
    print(f"✅ Image saved to {save_path}")


def resize_images_batch(image_paths, target_size=(224, 224)):
    """
    Resize multiple images
    
    Args:
        image_paths: List of image paths
        target_size: Target dimensions
    
    Returns:
        List of resized images
    """
    resized = []
    for img_path in image_paths:
        img = load_image(img_path, target_size=target_size, normalize=False)
        if img is not None:
            resized.append(img)
    
    return resized


def apply_augmentation_preview(image, save_path='augmentation_preview.png'):
    """
    Show augmentation effects on a single image
    
    Args:
        image: Input image (numpy array or path)
        save_path: Where to save preview
    """
    if isinstance(image, (str, Path)):
        image = load_image(image, normalize=True)
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    img = image.reshape((1,) + image.shape)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()
    
    # Original
    axes[0].imshow(image)
    axes[0].set_title('Original', fontweight='bold')
    axes[0].axis('off')
    
    # Augmented versions
    i = 1
    for batch in datagen.flow(img, batch_size=1):
        axes[i].imshow(batch[0])
        axes[i].set_title(f'Augmented {i}', fontweight='bold')
        axes[i].axis('off')
        i += 1
        if i >= 12:
            break
    
    plt.suptitle('Augmentation Preview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Augmentation preview saved to {save_path}")
    plt.close()


# ============================================================================
# DATA ANALYSIS UTILITIES
# ============================================================================

def analyze_dataset_distribution(data_path):
    """
    Analyze class distribution in dataset
    
    Args:
        data_path: Path to dataset directory
    
    Returns:
        DataFrame with class statistics
    """
    data_path = Path(data_path)
    class_stats = []
    
    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            class_stats.append({
                'class': class_dir.name,
                'count': len(images)
            })
    
    df = pd.DataFrame(class_stats)
    df['percentage'] = (df['count'] / df['count'].sum() * 100).round(2)
    df = df.sort_values('count', ascending=False).reset_index(drop=True)
    
    return df


def plot_class_distribution(df, save_path='class_distribution.png', top_n=20):
    """
    Plot class distribution
    
    Args:
        df: DataFrame with 'class' and 'count' columns
        save_path: Where to save plot
        top_n: Number of top classes to show
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_classes = df.head(top_n)
    ax.barh(range(len(top_classes)), top_classes['count'], color='steelblue')
    ax.set_yticks(range(len(top_classes)))
    ax.set_yticklabels(top_classes['class'], fontsize=10)
    ax.set_xlabel('Number of Images', fontsize=12)
    ax.set_title(f'Top {top_n} Classes by Image Count', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Class distribution plot saved to {save_path}")
    plt.close()


def check_image_quality(image_paths, sample_size=100):
    """
    Check for corrupted or invalid images
    
    Args:
        image_paths: List of image paths
        sample_size: Number of images to check
    
    Returns:
        Dictionary with quality statistics
    """
    sample = np.random.choice(image_paths, min(sample_size, len(image_paths)), replace=False)
    
    stats = {
        'total_checked': len(sample),
        'corrupted': 0,
        'valid': 0,
        'corrupted_files': []
    }
    
    for img_path in sample:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                stats['corrupted'] += 1
                stats['corrupted_files'].append(str(img_path))
            else:
                stats['valid'] += 1
        except Exception as e:
            stats['corrupted'] += 1
            stats['corrupted_files'].append(str(img_path))
    
    return stats


# ============================================================================
# MODEL EVALUATION UTILITIES
# ============================================================================

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history curves
    
    Args:
        history: Keras history object or dict
        save_path: Where to save plot
    """
    if hasattr(history, 'history'):
        history = history.history
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top-3 Accuracy (if available)
    if 'top_3_accuracy' in history:
        axes[1, 0].plot(history['top_3_accuracy'], label='Train', linewidth=2)
        axes[1, 0].plot(history['val_top_3_accuracy'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Top-3 Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-3 Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate (if available)
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], linewidth=2, color='orange')
        axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Training history plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix_custom(y_true, y_pred, class_names, 
                                 save_path='confusion_matrix.png',
                                 normalize=True):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Where to save plot
        normalize: Whether to normalize
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=False,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to {save_path}")
    plt.close()


def get_classification_report_df(y_true, y_pred, class_names):
    """
    Get classification report as DataFrame
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        DataFrame with classification metrics
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    df = pd.DataFrame(report).transpose()
    return df


def analyze_misclassifications(y_true, y_pred, class_names, top_n=10):
    """
    Find most confused class pairs
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        top_n: Number of top confused pairs
    
    Returns:
        DataFrame with confused pairs
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Remove diagonal (correct predictions)
    misclassified = cm.copy()
    np.fill_diagonal(misclassified, 0)
    
    # Find top confused pairs
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and misclassified[i, j] > 0:
                confused_pairs.append({
                    'true_class': class_names[i],
                    'predicted_as': class_names[j],
                    'count': int(misclassified[i, j])
                })
    
    df = pd.DataFrame(confused_pairs)
    df = df.sort_values('count', ascending=False).head(top_n)
    
    return df


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def get_model_summary(model):
    """
    Get detailed model summary
    
    Args:
        model: Keras model
    
    Returns:
        Dictionary with model statistics
    """
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    return {
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'non_trainable_params': int(non_trainable_params),
        'num_layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }


def compare_models(model_paths, test_generator):
    """
    Compare multiple trained models
    
    Args:
        model_paths: List of paths to saved models
        test_generator: Test data generator
    
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for model_path in model_paths:
        print(f"Evaluating {model_path}...")
        model = tf.keras.models.load_model(model_path)
        
        test_loss, test_acc = model.evaluate(test_generator, verbose=0)[:2]
        
        model_stats = get_model_summary(model)
        
        results.append({
            'model': Path(model_path).stem,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'params': model_stats['total_params']
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('test_accuracy', ascending=False)
    
    return df


# ============================================================================
# FILE I/O UTILITIES
# ============================================================================

def save_dict_to_json(data, filepath):
    """Save dictionary to JSON file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Data saved to {filepath}")


def load_json_to_dict(filepath):
    """Load JSON file to dictionary"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def save_dataframe_to_csv(df, filepath):
    """Save DataFrame to CSV"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(filepath, index=False)
    print(f"✅ DataFrame saved to {filepath}")


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def plot_sample_predictions(model, test_generator, class_names, 
                           num_samples=12, save_path='sample_predictions.png'):
    """
    Plot sample predictions with images
    
    Args:
        model: Trained model
        test_generator: Test data generator
        class_names: List of class names
        num_samples: Number of samples to show
        save_path: Where to save plot
    """
    # Get a batch of images
    images, labels = next(iter(test_generator))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Make predictions
    predictions = model.predict(images, verbose=0)
    
    # Plot
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx in range(min(num_samples, 12)):
        axes[idx].imshow(images[idx])
        
        true_label = class_names[np.argmax(labels[idx])]
        pred_label = class_names[np.argmax(predictions[idx])]
        confidence = np.max(predictions[idx]) * 100
        
        color = 'green' if true_label == pred_label else 'red'
        
        axes[idx].set_title(
            f'True: {true_label[:20]}\nPred: {pred_label[:20]}\n({confidence:.1f}%)',
            color=color,
            fontsize=9,
            fontweight='bold'
        )
        axes[idx].axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Sample predictions saved to {save_path}")
    plt.close()


def create_result_summary(metrics, save_path='result_summary.txt'):
    """
    Create a text summary of results
    
    Args:
        metrics: Dictionary with evaluation metrics
        save_path: Where to save summary
    """
    summary = f"""
{'='*70}
MODEL EVALUATION SUMMARY
{'='*70}

Performance Metrics:
  • Test Accuracy: {metrics.get('test_accuracy', 'N/A'):.4f}
  • Test Loss: {metrics.get('test_loss', 'N/A'):.4f}
  • Precision: {metrics.get('test_precision', 'N/A'):.4f}
  • Recall: {metrics.get('test_recall', 'N/A'):.4f}
  • F1-Score: {metrics.get('test_f1', 'N/A'):.4f}

Model Information:
  • Total Parameters: {metrics.get('total_params', 'N/A'):,}
  • Trainable Parameters: {metrics.get('trainable_params', 'N/A'):,}
  • Model Size: {metrics.get('model_size_mb', 'N/A'):.2f} MB

Training Details:
  • Epochs Trained: {metrics.get('epochs_trained', 'N/A')}
  • Best Epoch: {metrics.get('best_epoch', 'N/A')}
  • Training Time: {metrics.get('training_time', 'N/A')}

{'='*70}
"""
    
    with open(save_path, 'w') as f:
        f.write(summary)
    
    print(f"✅ Result summary saved to {save_path}")
    print(summary)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("UTILITY FUNCTIONS - EXAMPLES")
    print("=" * 70)
    
    # Example 1: Load and process image
    print("\n1. Image Processing Example")
    # img = load_image('example.jpg', target_size=(224, 224))
    # print(f"   Image shape: {img.shape}")
    
    # Example 2: Analyze dataset
    print("\n2. Dataset Analysis Example")
    # df = analyze_dataset_distribution('data/processed/train')
    # print(df.head())
    
    # Example 3: Plot training history
    print("\n3. Visualization Example")
    # history = load_json_to_dict('results/training_history.json')
    # plot_training_history(history)
    
    print("\n✅ Utility functions loaded and ready to use!")
    print("   Import with: from utils import *")