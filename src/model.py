 
"""
Model Architectures for Crop Disease Detection
Includes: Custom CNN, MobileNetV2, EfficientNetB0 with Transfer Learning
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import (
    MobileNetV2, 
    EfficientNetB0, 
    ResNet50,
    VGG16
)
from tensorflow.keras.regularizers import l2


def create_custom_cnn(input_shape=(224, 224, 3), num_classes=38, dropout_rate=0.5):
    """
    Custom CNN architecture from scratch
    Good baseline model for comparison
    
    Args:
        input_shape: Input image dimensions
        num_classes: Number of disease classes
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Classification head
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ], name='CustomCNN')
    
    return model


def create_mobilenet_v2(input_shape=(224, 224, 3), num_classes=38, 
                        trainable_layers=20, dropout_rate=0.5):
    """
    MobileNetV2 with Transfer Learning
    Lightweight and efficient - RECOMMENDED for production
    
    Args:
        input_shape: Input image dimensions
        num_classes: Number of disease classes
        trainable_layers: Number of last layers to fine-tune
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Keras model ready for training
    """
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification head
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='MobileNetV2_Transfer')
    
    # Option to unfreeze last N layers for fine-tuning
    if trainable_layers > 0:
        base_model.trainable = True
        # Freeze all layers except last N
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
    
    return model


def create_efficientnet_b0(input_shape=(224, 224, 3), num_classes=38, 
                           trainable_layers=30, dropout_rate=0.5):
    """
    EfficientNetB0 with Transfer Learning
    Better accuracy than MobileNet but slightly slower
    
    Args:
        input_shape: Input image dimensions
        num_classes: Number of disease classes
        trainable_layers: Number of last layers to fine-tune
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Keras model ready for training
    """
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    base_model.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='EfficientNetB0_Transfer')
    
    if trainable_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
    
    return model


def create_resnet50(input_shape=(224, 224, 3), num_classes=38, 
                    trainable_layers=40, dropout_rate=0.5):
    """
    ResNet50 with Transfer Learning
    Deeper model for better accuracy (more parameters)
    """
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    base_model.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='ResNet50_Transfer')
    
    if trainable_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
    
    return model


class ModelBuilder:
    """
    Unified interface for building different models
    """
    
    AVAILABLE_MODELS = {
        'custom_cnn': create_custom_cnn,
        'mobilenet_v2': create_mobilenet_v2,
        'efficientnet_b0': create_efficientnet_b0,
        'resnet50': create_resnet50
    }
    
    @staticmethod
    def build(model_name='mobilenet_v2', input_shape=(224, 224, 3), 
              num_classes=38, **kwargs):
        """
        Build model by name
        
        Args:
            model_name: One of 'custom_cnn', 'mobilenet_v2', 'efficientnet_b0', 'resnet50'
            input_shape: Input image dimensions
            num_classes: Number of classes
            **kwargs: Additional model-specific parameters
        
        Returns:
            Keras model
        """
        if model_name not in ModelBuilder.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not available. Choose from {list(ModelBuilder.AVAILABLE_MODELS.keys())}")
        
        model_fn = ModelBuilder.AVAILABLE_MODELS[model_name]
        model = model_fn(input_shape=input_shape, num_classes=num_classes, **kwargs)
        
        print(f"\n✅ Built model: {model_name}")
        print(f"   Total params: {model.count_params():,}")
        print(f"   Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        
        return model
    
    @staticmethod
    def compile_model(model, learning_rate=0.001, metrics=['accuracy']):
        """
        Compile model with optimizer and loss function
        
        Args:
            model: Keras model
            learning_rate: Initial learning rate
            metrics: List of metrics to track
        
        Returns:
            Compiled model
        """
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=metrics + [
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        print(f"\n✅ Model compiled with learning_rate={learning_rate}")
        return model
    
    @staticmethod
    def print_summary(model, save_path=None):
        """Print and optionally save model architecture"""
        model.summary()
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
            print(f"\n✅ Model summary saved to {save_path}")


# Utility function for loading trained model
def load_trained_model(model_path):
    """
    Load a trained model from disk
    
    Args:
        model_path: Path to saved model (.h5 or .keras)
    
    Returns:
        Loaded Keras model
    """
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded from {model_path}")
    return model


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("MODEL ARCHITECTURE EXAMPLES")
    print("=" * 70)
    
    # Example 1: Custom CNN
    print("\n1. Custom CNN Model")
    custom_model = ModelBuilder.build('custom_cnn', num_classes=38)
    ModelBuilder.compile_model(custom_model, learning_rate=0.001)
    
    # Example 2: MobileNetV2 (Recommended)
    print("\n2. MobileNetV2 Transfer Learning Model")
    mobilenet_model = ModelBuilder.build(
        'mobilenet_v2', 
        num_classes=38,
        trainable_layers=20,
        dropout_rate=0.5
    )
    ModelBuilder.compile_model(mobilenet_model, learning_rate=0.001)
    
    # Example 3: EfficientNetB0
    print("\n3. EfficientNetB0 Model")
    efficientnet_model = ModelBuilder.build(
        'efficientnet_b0',
        num_classes=38,
        trainable_layers=30
    )
    ModelBuilder.compile_model(efficientnet_model, learning_rate=0.001)
    
    # Print detailed summary
    print("\n" + "=" * 70)
    print("MobileNetV2 Detailed Summary")
    print("=" * 70)
    ModelBuilder.print_summary(mobilenet_model, save_path='models/model_summary.txt')