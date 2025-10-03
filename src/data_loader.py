"""
Data Loader and Preprocessing Pipeline for Crop Disease Detection
Handles PlantVillage dataset loading, augmentation, splitting, and corrupted image handling
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
        self.split_info = {'train': [], 'val': [], 'test': []}
    
    def _filter_corrupted_images(self, images):
        """Return only valid images that can be read by OpenCV"""
        valid_images = []
        for img_path in images:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    valid_images.append(img_path)
            except Exception:
                print(f"⚠️  Corrupted image skipped: {img_path}")
        return valid_images
    
    def analyze_dataset(self):
        print("📊 Analyzing dataset...")
        class_data = []
        total_images = 0
        
        for class_dir in sorted(self.raw_data_path.iterdir()):
            if class_dir.is_dir():
                images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG')) + \
                         list(class_dir.glob('*.png')) + list(class_dir.glob('*.PNG'))
                
                images = self._filter_corrupted_images(images)  # filter corrupted images
                
                class_name = class_dir.name
                num_images = len(images)
                total_images += num_images

                img_shape = (0, 0, 0)
                if images:
                    sample_img = cv2.imread(str(images[0]))
                    img_shape = sample_img.shape if sample_img is not None else (0, 0, 0)
                
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
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1"
        print(f"\n📂 Creating train/val/test split ({train_ratio}/{val_ratio}/{test_ratio})...")
        
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            split_dir.mkdir(parents=True, exist_ok=True)
        
        total_copied = 0
        
        for class_dir in tqdm(sorted(self.raw_data_path.iterdir()), desc="Processing classes"):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG')) + \
                     list(class_dir.glob('*.png')) + list(class_dir.glob('*.PNG'))
            
            images = self._filter_corrupted_images(images)
            
            if not images:
                print(f"⚠️  Warning: No valid images found in {class_name}")
                continue
            
            train_imgs, temp_imgs = train_test_split(images, test_size=(1 - train_ratio), random_state=self.seed)
            val_size = val_ratio / (val_ratio + test_ratio)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=(1 - val_size), random_state=self.seed)
            
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
        self._save_class_names()
        self._save_split_info()
    
    def _print_split_summary(self):
        for split in ['train', 'val', 'test']:
            split_path = self.processed_data_path / split
            total = sum([len(list(d.glob('*.*'))) for d in split_path.iterdir() if d.is_dir()])
            print(f"  {split.capitalize()}: {total} images")
    
    def _save_class_names(self):
        if self.class_names:
            class_names_path = self.processed_data_path / 'class_names.json'
            with open(class_names_path, 'w') as f:
                json.dump(self.class_names, f, indent=2)
            print(f"\n✅ Class names saved to {class_names_path}")
    
    def _save_split_info(self):
        split_csv_path = self.processed_data_path / 'split_info.csv'
        pd.DataFrame(dict([(k, pd.Series(v)) for k, v in self.split_info.items()])).to_csv(split_csv_path, index=False)
        print(f"✅ Split info saved to {split_csv_path}")
    
    def create_data_generators(self, batch_size=32, augment_train=True):
        print(f"\n🔄 Creating data generators (batch_size={batch_size})...")
        
        if augment_train:
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
        
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            self.train_dir, target_size=self.img_size, batch_size=batch_size,
            class_mode='categorical', shuffle=True, seed=self.seed
        )
        val_generator = val_test_datagen.flow_from_directory(
            self.val_dir, target_size=self.img_size, batch_size=batch_size,
            class_mode='categorical', shuffle=False
        )
        test_generator = val_test_datagen.flow_from_directory(
            self.test_dir, target_size=self.img_size, batch_size=batch_size,
            class_mode='categorical', shuffle=False
        )
        
        print(f"✅ Train samples: {train_generator.samples}")
        print(f"✅ Val samples: {val_generator.samples}")
        print(f"✅ Test samples: {test_generator.samples}")
        print(f"✅ Number of classes: {train_generator.num_classes}")
        
        return train_generator, val_generator, test_generator
    
    def compute_class_weights(self):
        from sklearn.utils.class_weight import compute_class_weight
        class_counts = {d.name: len(list(d.glob('*.*'))) for d in self.train_dir.iterdir() if d.is_dir()}
        class_indices = {name: idx for idx, name in enumerate(sorted(class_counts.keys()))}
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array(list(class_indices.values())),
            y=np.repeat(list(class_indices.values()), [class_counts[name] for name in sorted(class_counts.keys())])
        )
        class_weight_dict = {idx: weight for idx, weight in enumerate(class_weights)}
        print("\n⚖️  Class weights computed for imbalanced dataset")
        return class_weight_dict


# Utility functions remain the same
def load_and_preprocess_image(image_path, img_size=(224, 224)):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    return img


def visualize_augmentation(image_path, save_path='augmentation_examples.png'):
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
    
    for i, batch in enumerate(datagen.flow(img, batch_size=1)):
        axes[i].imshow(batch[0])
        axes[i].axis('off')
        axes[i].set_title(f'Augmented {i+1}')
        if i >= 7:
            break
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Augmentation examples saved to {save_path}")

def _process_and_copy_image(self, src_path, dest_path): # Corrected method definition
    """Read, convert to RGB, resize, and save image to destination"""
    try:
        img = cv2.imread(str(src_path))
        if img is None:
            print(f"⚠️ Corrupted image skipped: {src_path}")
            return False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        cv2.imwrite(str(dest_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        print(f"⚠️ Failed to process {src_path}: {e}")
        return False


# Example usage
if __name__ == "__main__":
    loader = PlantVillageDataLoader(
        raw_data_path=Path(r"C:/Users/anask/crop-disease-detection/data/raw/plantvillage-dataset/plantvillage dataset/color"),
        processed_data_path='data/processed',
        img_size=(224, 224)
    )
    
    df = loader.analyze_dataset()
    loader.create_train_val_test_split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train_gen, val_gen, test_gen = loader.create_data_generators(batch_size=32, augment_train=True)
    class_weights = loader.compute_class_weights()
    
    print("\n🎉 Data pipeline ready for training!")
