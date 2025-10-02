"""
Data Loader and Preprocessing Pipeline for Crop Disease Detection
Handles PlantVillage dataset loading, augmentation, and splitting
"""

import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tqdm import tqdm
import json


class PlantVillageDataLoader:
    """
    Complete data loading and preprocessing pipeline for PlantVillage dataset
    """
    
    def __init__(self, raw_data_path, processed_data_path, img_size=(224, 224), seed=42):
        """
        Args:
            raw_data_path: Path to raw PlantVillage dataset
            processed_data_path: Path to save processed/split data
            img_size: Target image size (height, width)
            seed: Random seed for reproducibility
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.img_size = img_size
        self.seed = seed
        
        # Create directory structure
        self.train_dir = self.processed_data_path / 'train'
        self.val_dir = self.processed_data_path / 'val'
        self.test_dir = self.processed_data_path / 'test'
        
        self.class_names = None
        self.class_distribution = None
        
    def analyze_dataset(self):
        """
        Perform initial dataset analysis
        Returns: DataFrame with class statistics
        """
        print("📊 Analyzing dataset...")
        
        class_data = []
        total_images = 0
        
        for class_dir in sorted(self.raw_data_path.iterdir()):
            if class_dir.is_dir():
                images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG')) + \
                         list(class_dir.glob('*.png')) + list(class_dir.glob('*.PNG'))
                
                class_name = class_dir.name
                num_images = len(images)
                total_images += num_images
                
                # Sample image for size analysis
                if images:
                    sample_img = cv2.imread(str(images[0]))
                    img_shape = sample_img.shape if sample_img is not None else (0, 0, 0)
                else:
                    img_shape = (0, 0, 0)
                
                class_data.append({
                    'class_name': class_name,
                    'num_images': num_images,
                    'sample_height': img_shape[0],
                    'sample_width': img_shape[1]
                })
        
        df = pd.DataFrame(class_data)
        df['percentage'] = (df['num_images'] / total_images * 100).round(2)
        
        self.class_names = df['class_name'].tolist()
        self.class_distribution = df
        
        print(f"\n✅ Found {len(self.class_names)} classes")
        print(f"✅ Total images: {total_images}")
        print(f"\n{'Class Distribution:'}")
        print(df.to_string(index=False))
        
        return df
    
    def create_train_val_test_split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split dataset into train/val/test with stratification
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1"
        
        print(f"\n📂 Creating train/val/test split ({train_ratio}/{val_ratio}/{test_ratio})...")
        
        # Create directories
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            split_dir.mkdir(parents=True, exist_ok=True)
        
        total_copied = 0
        
        for class_dir in tqdm(sorted(self.raw_data_path.iterdir()), desc="Processing classes"):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            
            # Get all images
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG')) + \
                     list(class_dir.glob('*.png')) + list(class_dir.glob('*.PNG'))
            
            if len(images) == 0:
                print(f"⚠️  Warning: No images found in {class_name}")
                continue
            
            # First split: train vs (val+test)
            train_imgs, temp_imgs = train_test_split(
                images, 
                test_size=(1 - train_ratio), 
                random_state=self.seed
            )
            
            # Second split: val vs test
            val_size = val_ratio / (val_ratio + test_ratio)
            val_imgs, test_imgs = train_test_split(
                temp_imgs, 
                test_size=(1 - val_size), 
                random_state=self.seed
            )
            
            # Copy files to respective directories
            for split_name, split_imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
                split_class_dir = self.processed_data_path / split_name / class_name
                split_class_dir.mkdir(parents=True, exist_ok=True)
                
                for img_path in split_imgs:
                    dest = split_class_dir / img_path.name
                    if not dest.exists():
                        shutil.copy2(img_path, dest)
                        total_copied += 1
        
        print(f"\n✅ Dataset split complete! Copied {total_copied} images")
        self._print_split_summary()
        
        # Save class names
        self._save_class_names()
    
    def _print_split_summary(self):
        """Print summary of train/val/test split"""
        for split in ['train', 'val', 'test']:
            split_path = self.processed_data_path / split
            total = sum([len(list(d.glob('*.*'))) for d in split_path.iterdir() if d.is_dir()])
            print(f"  {split.capitalize()}: {total} images")
    
    def _save_class_names(self):
        """Save class names to JSON for later use"""
        if self.class_names:
            class_names_path = self.processed_data_path / 'class_names.json'
            with open(class_names_path, 'w') as f:
                json.dump(self.class_names, f, indent=2)
            print(f"\n✅ Class names saved to {class_names_path}")
    
    def create_data_generators(self, batch_size=32, augment_train=True):
        """
        Create TensorFlow data generators with augmentation
        
        Args:
            batch_size: Batch size for training
            augment_train: Whether to apply augmentation to training data
            
        Returns:
            train_gen, val_gen, test_gen
        """
        print(f"\n🔄 Creating data generators (batch_size={batch_size})...")
        
        if augment_train:
            # Strong augmentation for training
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest',
                brightness_range=[0.8, 1.2]
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        # No augmentation for val/test
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=self.seed
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"✅ Train samples: {train_generator.samples}")
        print(f"✅ Val samples: {val_generator.samples}")
        print(f"✅ Test samples: {test_generator.samples}")
        print(f"✅ Number of classes: {train_generator.num_classes}")
        
        return train_generator, val_generator, test_generator
    
    def compute_class_weights(self):
        """
        Compute class weights for handling imbalanced dataset
        Returns: Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        # Count samples per class in training set
        class_counts = {}
        for class_dir in self.train_dir.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob('*.*')))
                class_counts[class_dir.name] = count
        
        # Get class indices
        class_indices = {name: idx for idx, name in enumerate(sorted(class_counts.keys()))}
        
        # Compute weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array(list(class_indices.values())),
            y=np.repeat(list(class_indices.values()), 
                       [class_counts[name] for name in sorted(class_counts.keys())])
        )
        
        class_weight_dict = {idx: weight for idx, weight in enumerate(class_weights)}
        
        print("\n⚖️  Class weights computed for imbalanced dataset")
        return class_weight_dict


# Utility functions
def load_and_preprocess_image(image_path, img_size=(224, 224)):
    """
    Load and preprocess a single image
    """
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    return img


def visualize_augmentation(image_path, save_path='augmentation_examples.png'):
    """
    Visualize augmentation effects on a sample image
    """
    import matplotlib.pyplot as plt
    
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    img = load_and_preprocess_image(image_path)
    img = img.reshape((1,) + img.shape)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    i = 0
    for batch in datagen.flow(img, batch_size=1):
        axes[i].imshow(batch[0])
        axes[i].axis('off')
        axes[i].set_title(f'Augmented {i+1}')
        i += 1
        if i >= 8:
            break
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Augmentation examples saved to {save_path}")


# Example usage
if __name__ == "__main__":
    # Initialize data loader
    loader = PlantVillageDataLoader(
        raw_data_path='data/raw/PlantVillage',
        processed_data_path='data/processed',
        img_size=(224, 224)
    )
    
    # Step 1: Analyze dataset
    df = loader.analyze_dataset()
    
    # Step 2: Create train/val/test split
    loader.create_train_val_test_split(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Step 3: Create data generators
    train_gen, val_gen, test_gen = loader.create_data_generators(
        batch_size=32,
        augment_train=True
    )
    
    # Step 4: Compute class weights (for imbalanced data)
    class_weights = loader.compute_class_weights()
    
    print("\n🎉 Data pipeline ready for training!")