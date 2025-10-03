 
"""
Complete Training Pipeline for Crop Disease Detection
Includes: Training, Validation, Callbacks, Metrics, Model Saving
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
    TensorBoard, CSVLogger
)
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support
)

# Import custom modules
from model import ModelBuilder
from data_loader import PlantVillageDataLoader


class CropDiseaseTrainer:
    """
    Complete training pipeline with evaluation and visualization
    """
    
    def __init__(self, config):
        """
        Args:
            config: Dictionary with training configuration
        """
        self.config = config
        self.model = None
        self.history = None
        self.class_names = None
        
        # Create directories
        self.model_dir = Path(config['model_dir'])
        self.results_dir = Path(config['results_dir'])
        self.logs_dir = Path(config['logs_dir'])
        
        for dir_path in [self.model_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(config['seed'])
        tf.random.set_seed(config['seed'])
        
        print("=" * 70)
        print("CROP DISEASE DETECTION - TRAINING PIPELINE")
        print("=" * 70)
        print(f"\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    def prepare_data(self):
        """Load and prepare data generators"""
        print("\n" + "=" * 70)
        print("DATA PREPARATION")
        print("=" * 70)
        
        loader = PlantVillageDataLoader(
            raw_data_path=self.config['data_path'],
            processed_data_path=self.config['processed_data_path'],
            img_size=tuple(self.config['img_size']),
            seed=self.config['seed']
        )
        
        # Create data generators
        self.train_gen, self.val_gen, self.test_gen = loader.create_data_generators(
            batch_size=self.config['batch_size'],
            augment_train=self.config['augment_data']
        )
        
        # Load class names
        class_names_path = Path(self.config['processed_data_path']) / 'class_names.json'
        if class_names_path.exists():
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
        else:
            self.class_names = list(self.train_gen.class_indices.keys())
        
        # Compute class weights for imbalanced data
        if self.config['use_class_weights']:
            self.class_weights = loader.compute_class_weights()
        else:
            self.class_weights = None
        
        print(f"\n✅ Data preparation complete!")
        print(f"   Classes: {len(self.class_names)}")
        print(f"   Train batches: {len(self.train_gen)}")
        print(f"   Val batches: {len(self.val_gen)}")
        print(f"   Test batches: {len(self.test_gen)}")
    
    def build_model(self):
        """Build and compile model"""
        print("\n" + "=" * 70)
        print("MODEL BUILDING")
        print("=" * 70)
        
        self.model = ModelBuilder.build(
            model_name=self.config['model_name'],
            input_shape=tuple(self.config['img_size']) + (3,),
            num_classes=len(self.class_names),
            trainable_layers=self.config['trainable_layers'],
            dropout_rate=self.config['dropout_rate']
        )
        
        ModelBuilder.compile_model(
            self.model, 
            learning_rate=self.config['learning_rate']
        )
        
        # Save model architecture
        ModelBuilder.print_summary(
            self.model, 
            save_path=self.model_dir / 'model_architecture.txt'
        )
    
    def get_callbacks(self):
        """Create training callbacks"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=str(self.model_dir / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=str(self.logs_dir / timestamp),
                histogram_freq=1,
                write_graph=True
            ),
            
            # CSV logging
            CSVLogger(
                filename=str(self.results_dir / f'training_log_{timestamp}.csv'),
                separator=',',
                append=False
            )
        ]
        
        return callbacks
    
    def train(self):
        """Execute training"""
        print("\n" + "=" * 70)
        print("TRAINING")
        print("=" * 70)
        
        callbacks = self.get_callbacks()
        
        # Calculate steps per epoch
        steps_per_epoch = len(self.train_gen)
        validation_steps = len(self.val_gen)
        
        print(f"\nStarting training for {self.config['epochs']} epochs...")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        
        # Train model
        self.history = self.model.fit(
            self.train_gen,
            epochs=self.config['epochs'],
            validation_data=self.val_gen,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        print("\n✅ Training complete!")
        
        # Save final model
        final_model_path = self.model_dir / 'final_model.h5'
        self.model.save(final_model_path)
        print(f"✅ Final model saved to {final_model_path}")
        
        # Save training history
        self.save_training_history()
    
    def save_training_history(self):
        """Save training history to JSON"""
        history_dict = {k: [float(v) for v in vals] for k, vals in self.history.history.items()}
        
        history_path = self.results_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"✅ Training history saved to {history_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        print("\n" + "=" * 70)
        print("PLOTTING TRAINING CURVES")
        print("=" * 70)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-3 Accuracy
        if 'top_3_accuracy' in self.history.history:
            axes[1, 0].plot(self.history.history['top_3_accuracy'], label='Train Top-3', linewidth=2)
            axes[1, 0].plot(self.history.history['val_top_3_accuracy'], label='Val Top-3', linewidth=2)
            axes[1, 0].set_title('Top-3 Accuracy', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-3 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'], linewidth=2, color='orange')
            axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.results_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✅ Training curves saved to {plot_path}")
        plt.close()
    
    def evaluate_model(self):
        """Evaluate model on test set"""
        print("\n" + "=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_acc, test_top3, test_precision, test_recall = self.model.evaluate(
            self.test_gen,
            verbose=1
        )
        
        print(f"\n📊 Test Results:")
        print(f"   Loss: {test_loss:.4f}")
        print(f"   Accuracy: {test_acc:.4f}")
        print(f"   Top-3 Accuracy: {test_top3:.4f}")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall: {test_recall:.4f}")
        
        # Get predictions
        print("\nGenerating predictions...")
        y_pred_probs = self.model.predict(self.test_gen, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = self.test_gen.classes
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Save classification report
        report_df = pd.DataFrame(report).transpose()
        report_path = self.results_dir / 'classification_report.csv'
        report_df.to_csv(report_path)
        print(f"\n✅ Classification report saved to {report_path}")
        
        # Print summary
        print("\n" + classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0))
        
        # Confusion Matrix
        self.plot_confusion_matrix(y_true, y_pred)
        
        # Per-class accuracy
        self.plot_per_class_accuracy(report)
        
        # Save evaluation metrics
        eval_metrics = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_top3_accuracy': float(test_top3),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'timestamp': datetime.now().isoformat()
        }
        
        metrics_path = self.results_dir / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        print(f"✅ Evaluation metrics saved to {metrics_path}")
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        plt.figure(figsize=(20, 18))
        sns.heatmap(
            cm_normalized, 
            annot=False,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Normalized Count'}
        )
        plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        cm_path = self.results_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to {cm_path}")
        plt.close()
    
    def plot_per_class_accuracy(self, report):
        """Plot per-class accuracy"""
        # Extract per-class metrics
        class_metrics = []
        for class_name in self.class_names:
            if class_name in report:
                class_metrics.append({
                    'class': class_name,
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1-score': report[class_name]['f1-score']
                })
        
        df_metrics = pd.DataFrame(class_metrics)
        df_metrics = df_metrics.sort_values('f1-score', ascending=True)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, max(10, len(self.class_names) * 0.3)))
        
        x = np.arange(len(df_metrics))
        width = 0.25
        
        ax.barh(x - width, df_metrics['precision'], width, label='Precision', color='steelblue')
        ax.barh(x, df_metrics['recall'], width, label='Recall', color='coral')
        ax.barh(x + width, df_metrics['f1-score'], width, label='F1-Score', color='mediumseagreen')
        
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_yticks(x)
        ax.set_yticklabels(df_metrics['class'], fontsize=8)
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        metrics_path = self.results_dir / 'per_class_metrics.png'
        plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
        print(f"✅ Per-class metrics saved to {metrics_path}")
        plt.close()
    
    def run_full_pipeline(self):
        """Execute complete training pipeline"""
        self.prepare_data()
        self.build_model()
        self.train()
        self.plot_training_curves()
        self.evaluate_model()
        
        print("\n" + "=" * 70)
        print("🎉 TRAINING PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"\nResults saved to: {self.results_dir}")
        print(f"Model saved to: {self.model_dir}")
        print(f"Logs saved to: {self.logs_dir}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Crop Disease Detection Model')
    
    parser.add_argument('--model', type=str, default='mobilenet_v2',
                       choices=['custom_cnn', 'mobilenet_v2', 'efficientnet_b0', 'resnet50'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224], help='Image size (H W)')
    parser.add_argument('--data_path', type=str, default='data/processed',
                       help='Path to processed data')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Training configuration
    config = {
        'model_name': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'img_size': args.img_size,
        'data_path': 'data/raw/PlantVillage',
        'processed_data_path': args.data_path,
        'model_dir': 'models',
        'results_dir': 'results',
        'logs_dir': 'logs',
        'augment_data': True,
        'use_class_weights': True,
        'trainable_layers': 20,
        'dropout_rate': 0.5,
        'early_stopping_patience': 7,
        'reduce_lr_patience': 3,
        'seed': 42
    }
    
    # Initialize and run trainer
    trainer = CropDiseaseTrainer(config)
    trainer.run_full_pipeline()