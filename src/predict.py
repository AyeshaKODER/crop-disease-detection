"""
Standalone Prediction Script for Crop Disease Detection
Usage: python predict.py --image path/to/image.jpg --model models/best_model.h5
"""

import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf


class DiseasePredictor:
    """
    Standalone predictor for crop disease detection
    """
    
    def __init__(self, model_path, class_names_path='data/processed/class_names.json'):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to trained model (.h5 file)
            class_names_path: Path to class names JSON
        """
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")
        
        # Load class names
        if Path(class_names_path).exists():
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
        else:
            print(f"⚠️  Class names not found at {class_names_path}")
            self.class_names = [f"Class_{i}" for i in range(38)]
        
        print(f"✅ Loaded {len(self.class_names)} class names")
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """
        Preprocess image for prediction
        
        Args:
            image_path: Path to image file
            target_size: Target dimensions
        
        Returns:
            Preprocessed numpy array
        """
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize(target_size)
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_path, top_k=3):
        """
        Predict disease from image
        
        Args:
            image_path: Path to leaf image
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        
        # Get top prediction
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx] * 100
        predicted_class = self.class_names[predicted_idx]
        
        # Get top-k predictions
        top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]
        top_k_predictions = [
            {
                'rank': i + 1,
                'class': self.class_names[idx],
                'confidence': float(predictions[0][idx] * 100)
            }
            for i, idx in enumerate(top_k_indices)
        ]
        
        # Prepare result
        result = {
            'image_path': str(image_path),
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'top_predictions': top_k_predictions
        }
        
        return result
    
    def predict_batch(self, image_paths, top_k=3):
        """
        Predict diseases for multiple images
        
        Args:
            image_paths: List of image paths
            top_k: Number of top predictions per image
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        print(f"\nProcessing {len(image_paths)} images...")
        for i, img_path in enumerate(image_paths, 1):
            print(f"[{i}/{len(image_paths)}] Processing {img_path}...")
            try:
                result = self.predict(img_path, top_k=top_k)
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
                results.append({
                    'image_path': str(img_path),
                    'error': str(e)
                })
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction with image
        
        Args:
            image_path: Path to image
            save_path: Optional path to save visualization
        """
        import matplotlib.pyplot as plt
        
        # Get prediction
        result = self.predict(image_path)
        
        # Load original image
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Display image
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('Input Image', fontsize=14, fontweight='bold')
        
        # Display prediction
        pred_text = f"Prediction:\n{result['predicted_class']}\n\n"
        pred_text += f"Confidence: {result['confidence']:.2f}%\n\n"
        pred_text += "Top 3 Predictions:\n"
        for pred in result['top_predictions']:
            pred_text += f"{pred['rank']}. {pred['class']}\n"
            pred_text += f"   ({pred['confidence']:.2f}%)\n"
        
        ax2.text(0.1, 0.5, pred_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.axis('off')
        ax2.set_title('Prediction Results', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main function with CLI"""
    parser = argparse.ArgumentParser(description='Crop Disease Prediction')
    
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--model', type=str, default='models/best_model.h5',
                       help='Path to trained model')
    parser.add_argument('--class_names', type=str, default='data/processed/class_names.json',
                       help='Path to class names JSON')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top predictions to show')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize prediction')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save visualization')
    parser.add_argument('--batch', action='store_true',
                       help='Process directory of images')
    parser.add_argument('--save_json', type=str, default=None,
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = DiseasePredictor(args.model, args.class_names)
    
    print("\n" + "="*70)
    print("CROP DISEASE PREDICTION")
    print("="*70)
    
    # Process single image or batch
    if args.batch:
        # Process directory
        image_dir = Path(args.image)
        if not image_dir.is_dir():
            print(f"❌ {args.image} is not a directory")
            return
        
        image_paths = list(image_dir.glob('*.jpg')) + \
                     list(image_dir.glob('*.jpeg')) + \
                     list(image_dir.glob('*.png'))
        
        if not image_paths:
            print(f"❌ No images found in {args.image}")
            return
        
        results = predictor.predict_batch(image_paths, top_k=args.top_k)
        
        # Print summary
        print("\n" + "="*70)
        print("BATCH PREDICTION SUMMARY")
        print("="*70)
        for result in results:
            if 'error' not in result:
                print(f"\n{Path(result['image_path']).name}:")
                print(f"  → {result['predicted_class']}")
                print(f"  → Confidence: {result['confidence']:.2f}%")
        
        # Save to JSON if requested
        if args.save_json:
            with open(args.save_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✅ Results saved to {args.save_json}")
    
    else:
        # Process single image
        if not Path(args.image).exists():
            print(f"❌ Image not found: {args.image}")
            return
        
        result = predictor.predict(args.image, top_k=args.top_k)
        
        # Print results
        print(f"\n📸 Image: {result['image_path']}")
        print(f"\n🎯 Prediction: {result['predicted_class']}")
        print(f"📊 Confidence: {result['confidence']:.2f}%")
        
        print(f"\n🔝 Top {args.top_k} Predictions:")
        for pred in result['top_predictions']:
            print(f"  {pred['rank']}. {pred['class']} ({pred['confidence']:.2f}%)")
        
        # Visualize if requested
        if args.visualize:
            predictor.visualize_prediction(args.image, save_path=args.output)
        
        # Save to JSON if requested
        if args.save_json:
            with open(args.save_json, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n✅ Results saved to {args.save_json}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main() 
"""
Standalone Prediction Script for Crop Disease Detection
Usage: python predict.py --image path/to/image.jpg --model models/best_model.h5
"""

import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf


class DiseasePredictor:
    """
    Standalone predictor for crop disease detection
    """
    
    def __init__(self, model_path, class_names_path='data/processed/class_names.json'):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to trained model (.h5 file)
            class_names_path: Path to class names JSON
        """
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")
        
        # Load class names
        if Path(class_names_path).exists():
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
        else:
            print(f"⚠️  Class names not found at {class_names_path}")
            self.class_names = [f"Class_{i}" for i in range(38)]
        
        print(f"✅ Loaded {len(self.class_names)} class names")
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """
        Preprocess image for prediction
        
        Args:
            image_path: Path to image file
            target_size: Target dimensions
        
        Returns:
            Preprocessed numpy array
        """
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize(target_size)
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_path, top_k=3):
        """
        Predict disease from image
        
        Args:
            image_path: Path to leaf image
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        
        # Get top prediction
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx] * 100
        predicted_class = self.class_names[predicted_idx]
        
        # Get top-k predictions
        top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]
        top_k_predictions = [
            {
                'rank': i + 1,
                'class': self.class_names[idx],
                'confidence': float(predictions[0][idx] * 100)
            }
            for i, idx in enumerate(top_k_indices)
        ]
        
        # Prepare result
        result = {
            'image_path': str(image_path),
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'top_predictions': top_k_predictions
        }
        
        return result
    
    def predict_batch(self, image_paths, top_k=3):
        """
        Predict diseases for multiple images
        
        Args:
            image_paths: List of image paths
            top_k: Number of top predictions per image
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        print(f"\nProcessing {len(image_paths)} images...")
        for i, img_path in enumerate(image_paths, 1):
            print(f"[{i}/{len(image_paths)}] Processing {img_path}...")
            try:
                result = self.predict(img_path, top_k=top_k)
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
                results.append({
                    'image_path': str(img_path),
                    'error': str(e)
                })
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction with image
        
        Args:
            image_path: Path to image
            save_path: Optional path to save visualization
        """
        import matplotlib.pyplot as plt
        
        # Get prediction
        result = self.predict(image_path)
        
        # Load original image
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Display image
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('Input Image', fontsize=14, fontweight='bold')
        
        # Display prediction
        pred_text = f"Prediction:\n{result['predicted_class']}\n\n"
        pred_text += f"Confidence: {result['confidence']:.2f}%\n\n"
        pred_text += "Top 3 Predictions:\n"
        for pred in result['top_predictions']:
            pred_text += f"{pred['rank']}. {pred['class']}\n"
            pred_text += f"   ({pred['confidence']:.2f}%)\n"
        
        ax2.text(0.1, 0.5, pred_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.axis('off')
        ax2.set_title('Prediction Results', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main function with CLI"""
    parser = argparse.ArgumentParser(description='Crop Disease Prediction')
    
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--model', type=str, default='models/best_model.h5',
                       help='Path to trained model')
    parser.add_argument('--class_names', type=str, default='data/processed/class_names.json',
                       help='Path to class names JSON')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top predictions to show')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize prediction')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save visualization')
    parser.add_argument('--batch', action='store_true',
                       help='Process directory of images')
    parser.add_argument('--save_json', type=str, default=None,
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = DiseasePredictor(args.model, args.class_names)
    
    print("\n" + "="*70)
    print("CROP DISEASE PREDICTION")
    print("="*70)
    
    # Process single image or batch
    if args.batch:
        # Process directory
        image_dir = Path(args.image)
        if not image_dir.is_dir():
            print(f"❌ {args.image} is not a directory")
            return
        
        image_paths = list(image_dir.glob('*.jpg')) + \
                     list(image_dir.glob('*.jpeg')) + \
                     list(image_dir.glob('*.png'))
        
        if not image_paths:
            print(f"❌ No images found in {args.image}")
            return
        
        results = predictor.predict_batch(image_paths, top_k=args.top_k)
        
        # Print summary
        print("\n" + "="*70)
        print("BATCH PREDICTION SUMMARY")
        print("="*70)
        for result in results:
            if 'error' not in result:
                print(f"\n{Path(result['image_path']).name}:")
                print(f"  → {result['predicted_class']}")
                print(f"  → Confidence: {result['confidence']:.2f}%")
        
        # Save to JSON if requested
        if args.save_json:
            with open(args.save_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✅ Results saved to {args.save_json}")
    
    else:
        # Process single image
        if not Path(args.image).exists():
            print(f"❌ Image not found: {args.image}")
            return
        
        result = predictor.predict(args.image, top_k=args.top_k)
        
        # Print results
        print(f"\n📸 Image: {result['image_path']}")
        print(f"\n🎯 Prediction: {result['predicted_class']}")
        print(f"📊 Confidence: {result['confidence']:.2f}%")
        
        print(f"\n🔝 Top {args.top_k} Predictions:")
        for pred in result['top_predictions']:
            print(f"  {pred['rank']}. {pred['class']} ({pred['confidence']:.2f}%)")
        
        # Visualize if requested
        if args.visualize:
            predictor.visualize_prediction(args.image, save_path=args.output)
        
        # Save to JSON if requested
        if args.save_json:
            with open(args.save_json, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n✅ Results saved to {args.save_json}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()