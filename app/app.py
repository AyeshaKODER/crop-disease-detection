"""
Streamlit Web Application for Crop Disease Detection
Farmer-friendly interface with comprehensive treatment information
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import json
from pathlib import Path
import tensorflow as tf
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 25px;
        border-radius: 12px;
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 6px solid #4CAF50;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .info-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 15px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .warning-card {
        background-color: #FFF3E0;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #FF9800;
        margin: 15px 0;
    }
    .treatment-card {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .severity-high {
        background-color: #FFEBEE;
        border-left: 4px solid #F44336;
    }
    .severity-moderate {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
    }
    .severity-low {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path='models/saved_models/best_model.h5'):
    """Load trained model"""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_class_names(class_names_path='src/data/processed/class_names.json'):
    """Load class names"""
    try:
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        return class_names
    except Exception as e:
        st.warning(f"Could not load class names: {e}")
        return None


def get_comprehensive_disease_info(disease_name):
    """
    Get comprehensive disease information including crop, disease, treatment, and prevention
    """
    # Parse disease name (format: Crop___Disease)
    parts = disease_name.split('___')
    if len(parts) == 2:
        crop = parts[0].replace('_', ' ')
        disease = parts[1].replace('_', ' ')
    else:
        crop = "Unknown"
        disease = disease_name.replace('_', ' ')
    
    # Comprehensive disease database
    disease_database = {
        'Apple___Apple_scab': {
            'crop': 'Apple',
            'disease': 'Apple Scab',
            'severity': 'High',
            'description': 'Apple scab is a fungal disease caused by Venturia inaequalis. It appears as olive-green to brown lesions on leaves and fruit, causing premature leaf drop and fruit deformity.',
            'symptoms': [
                'Olive-green to brown spots on leaves',
                'Velvety appearance on infected areas',
                'Premature leaf drop',
                'Scabby lesions on fruit',
                'Cracked or deformed fruit'
            ],
            'causes': [
                'Wet, cool spring weather',
                'Poor air circulation',
                'Infected fallen leaves from previous season',
                'Overhead irrigation'
            ],
            'treatment': [
                'Apply fungicides (Captan, Mancozeb) at bud break',
                'Remove and destroy infected leaves and fruit',
                'Prune trees to improve air circulation',
                'Apply dormant oil spray in early spring'
            ],
            'prevention': [
                'Plant resistant apple varieties (Liberty, Freedom, Enterprise)',
                'Rake and remove fallen leaves in autumn',
                'Avoid overhead watering',
                'Maintain proper spacing between trees',
                'Apply preventive fungicide sprays during wet periods'
            ],
            'organic_solutions': [
                'Neem oil spray every 7-14 days',
                'Copper-based fungicides',
                'Baking soda solution (1 tbsp per gallon of water)',
                'Sulfur-based fungicides'
            ],
            'economic_impact': 'Can reduce yield by 70% if untreated',
            'best_time_to_treat': 'Early spring before symptoms appear'
        },
        'Tomato___Late_blight': {
            'crop': 'Tomato',
            'disease': 'Late Blight',
            'severity': 'Critical',
            'description': 'Late blight is a devastating disease caused by Phytophthora infestans. It can destroy entire tomato crops within days during favorable weather conditions.',
            'symptoms': [
                'Dark brown to black lesions on leaves',
                'White fuzzy growth on undersides of leaves',
                'Brown streaks on stems',
                'Firm brown spots on fruit',
                'Rapid plant collapse'
            ],
            'causes': [
                'Cool, wet weather (15-25°C with high humidity)',
                'Infected seed potatoes or transplants',
                'Wind-borne spores from nearby infected plants',
                'Excessive moisture on leaves'
            ],
            'treatment': [
                'Remove and destroy all infected plants immediately',
                'Apply copper-based fungicides every 5-7 days',
                'Use systemic fungicides (Chlorothalonil, Mancozeb)',
                'Improve drainage and air circulation'
            ],
            'prevention': [
                'Plant certified disease-free seedlings',
                'Use drip irrigation instead of overhead watering',
                'Space plants properly (2-3 feet apart)',
                'Apply preventive fungicide before disease appears',
                'Remove volunteer potato and tomato plants',
                'Choose resistant varieties (Mountain Magic, Defiant PHR)'
            ],
            'organic_solutions': [
                'Copper fungicides (Bordeaux mixture)',
                'Destroy infected plants completely',
                'No organic cure once established',
                'Focus on prevention'
            ],
            'economic_impact': 'Can cause 100% crop loss within 2 weeks',
            'best_time_to_treat': 'Prevention is critical - treat at first sign'
        },
        'Potato___Early_blight': {
            'crop': 'Potato',
            'disease': 'Early Blight',
            'severity': 'Moderate',
            'description': 'Early blight is caused by Alternaria solani fungus. It typically affects older leaves first and can reduce yield significantly.',
            'symptoms': [
                'Dark brown spots with concentric rings (target pattern)',
                'Yellow halo around spots',
                'Premature leaf drop starting from bottom leaves',
                'Brown lesions on tubers'
            ],
            'causes': [
                'Warm, humid weather',
                'Plant stress from drought or nutrient deficiency',
                'Infected seeds or soil',
                'Overhead irrigation'
            ],
            'treatment': [
                'Apply fungicides containing chlorothalonil or mancozeb',
                'Remove infected lower leaves',
                'Ensure adequate nutrition (nitrogen)',
                'Water at base of plants'
            ],
            'prevention': [
                'Rotate crops (3-4 year rotation)',
                'Use certified disease-free seed potatoes',
                'Mulch to prevent soil splash',
                'Maintain plant vigor with proper fertilization',
                'Avoid overhead irrigation'
            ],
            'organic_solutions': [
                'Copper fungicides',
                'Neem oil application',
                'Compost tea spray',
                'Remove infected plant parts'
            ],
            'economic_impact': 'Can reduce yield by 20-30%',
            'best_time_to_treat': 'Early season before flowering'
        }
    }
    
    # Return specific info or default
    if disease_name in disease_database:
        return disease_database[disease_name]
    else:
        # Generic info for unlisted diseases
        return {
            'crop': crop,
            'disease': disease,
            'severity': 'Moderate',
            'description': f'{disease} affecting {crop} plants. Consult local agricultural extension for specific treatment.',
            'symptoms': ['Consult agricultural expert for specific symptoms'],
            'causes': ['Various environmental and pathogenic factors'],
            'treatment': ['Consult local agricultural extension office', 'Consider fungicide or appropriate treatment', 'Remove severely infected plants'],
            'prevention': ['Practice crop rotation', 'Use disease-resistant varieties', 'Maintain good plant hygiene', 'Ensure proper spacing and air circulation'],
            'organic_solutions': ['Neem oil', 'Copper-based fungicides', 'Proper sanitation'],
            'economic_impact': 'Varies by severity',
            'best_time_to_treat': 'Early detection is key'
        }


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess uploaded image"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_disease(model, image, class_names):
    """Make prediction"""
    predictions = model.predict(image, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100
    predicted_class = class_names[predicted_idx]
    
    top_5_idx = np.argsort(predictions[0])[-5:][::-1]
    top_5_predictions = [
        {'class': class_names[idx], 'confidence': predictions[0][idx] * 100}
        for idx in top_5_idx
    ]
    
    return predicted_class, confidence, top_5_predictions, predictions[0]


def main():
    """Main application"""
    
    # Header
    st.markdown('<p class="main-header">🌾 Crop Disease Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Plant Disease Diagnosis for Farmers</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/AyeshaKODER/crop-disease-detection/main/app/static/leaf.png", 
                 use_container_width=True)
        
        st.markdown("### About This Tool")
        st.info(
            """
            This AI tool helps farmers identify crop diseases quickly and accurately.
            
            **What it does:**
            - Identifies 38 different crop diseases
            - 96%+ accuracy
            - Provides treatment recommendations
            - Suggests prevention methods
            """
        )
        
        st.markdown("### How to Use")
        st.markdown("""
        1. Take a clear photo of the affected leaf
        2. Upload the image using the button below
        3. Wait 2-3 seconds for analysis
        4. Read the disease information and treatment
        5. Follow prevention tips for future
        """)
        
        st.markdown("---")
        st.markdown("### Supported Crops")
        st.markdown("""
        - Apple, Blueberry, Cherry
        - Corn, Grape, Peach
        - Pepper, Potato, Strawberry
        - Tomato, and more
        """)
        
        st.markdown("---")
        st.warning("**Note:** This is an AI assistant. For serious infections, consult your local agricultural officer.")
    
    # Load model
    with st.spinner("Loading AI model..."):
        model = load_model()
        class_names = load_class_names()
    
    if model is None or class_names is None:
        st.error("Model could not be loaded. Please check installation.")
        return
    
    st.success("AI Model Ready!")
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload Leaf Image (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Take a clear photo of the infected leaf in good lighting"
    )
    
    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📸 Your Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption="Uploaded Leaf Image")
            
            st.info(f"**Image Size:** {image.size[0]} x {image.size[1]} pixels")
        
        with col2:
            st.markdown("### 🔬 AI Analysis")
            
            with st.spinner("Analyzing image... Please wait"):
                processed_image = preprocess_image(image)
                predicted_class, confidence, top_5, all_predictions = predict_disease(
                    model, processed_image, class_names
                )
                
                disease_info = get_comprehensive_disease_info(predicted_class)
                
                # Prediction box
                st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style="color: #1B5E20; margin: 0;">🎯 Detection Result</h2>
                        <h3 style="margin-top: 15px; color: #1B5E20; font-weight: bold;">
                            Crop: {disease_info['crop']}
                        </h3>
                        <h3 style="margin-top: 5px; color: #C62828; font-weight: bold;">
                            Disease: {disease_info['disease']}
                        </h3>
                        <p style="font-size: 1.3rem; font-weight: bold; color: #1976D2; margin-top: 10px;">
                            Confidence: {confidence:.1f}%
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Confidence indicator
                if confidence >= 90:
                    st.success("✅ High Confidence - Reliable detection")
                elif confidence >= 70:
                    st.warning("⚠️ Moderate Confidence - Consider expert consultation")
                else:
                    st.error("❌ Low Confidence - Please upload a clearer image")
        
        # Detailed Information Sections
        st.markdown("---")
        st.markdown("## 📋 Complete Disease Information")
        
        # Severity Alert
        severity_class = f"severity-{disease_info['severity'].lower()}"
        severity_color = {'Critical': '#D32F2F', 'High': '#F57C00', 'Moderate': '#FFA726', 'Low': '#66BB6A'}
        
        st.markdown(f"""
            <div class="info-card {severity_class}">
                <h3 style="color: {severity_color.get(disease_info['severity'], '#FF9800')}; margin: 0;">
                    ⚠️ Severity Level: {disease_info['severity']}
                </h3>
                <p style="color: #333; margin-top: 10px; font-size: 1.1rem;">
                    <strong>Economic Impact:</strong> {disease_info['economic_impact']}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Disease Description
        st.markdown("### 📖 What is this disease?")
        st.markdown(f"""
            <div class="info-card">
                <p style="color: #333; font-size: 1.05rem; line-height: 1.6;">
                    {disease_info['description']}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Symptoms
        st.markdown("### 🔍 Symptoms to Look For")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="treatment-card">', unsafe_allow_html=True)
            st.markdown("**Visible Signs:**")
            for symptom in disease_info['symptoms']:
                st.markdown(f"- {symptom}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="treatment-card">', unsafe_allow_html=True)
            st.markdown("**Common Causes:**")
            for cause in disease_info['causes']:
                st.markdown(f"- {cause}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Treatment Section
        st.markdown("---")
        st.markdown("## 💊 Treatment Recommendations")
        
        st.info(f"**⏰ Best Time to Treat:** {disease_info['best_time_to_treat']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧪 Chemical Treatment")
            st.markdown('<div class="treatment-card">', unsafe_allow_html=True)
            for idx, treatment in enumerate(disease_info['treatment'], 1):
                st.markdown(f"**Step {idx}:** {treatment}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🌿 Organic Solutions")
            st.markdown('<div class="treatment-card">', unsafe_allow_html=True)
            for idx, solution in enumerate(disease_info['organic_solutions'], 1):
                st.markdown(f"**Option {idx}:** {solution}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Prevention
        st.markdown("---")
        st.markdown("## 🛡️ Prevention for Future")
        st.markdown('<div class="warning-card">', unsafe_allow_html=True)
        st.markdown("**Follow these steps to prevent this disease:**")
        for idx, prevention in enumerate(disease_info['prevention'], 1):
            st.markdown(f"{idx}. {prevention}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Top 5 alternative possibilities
        st.markdown("---")
        st.markdown("### 🔬 Alternative Possibilities")
        
        df_top5 = pd.DataFrame(top_5)
        df_top5['class'] = df_top5['class'].apply(lambda x: x.replace('___', ' - ').replace('_', ' '))
        df_top5['confidence'] = df_top5['confidence'].round(2)
        
        fig = px.bar(
            df_top5,
            x='confidence',
            y='class',
            orientation='h',
            labels={'confidence': 'Confidence (%)', 'class': 'Disease'},
            title='Top 5 Possible Diseases',
            color='confidence',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Download report
        st.markdown("---")
        if st.button("📥 Download Detection Report", type="primary"):
            report = f"""
CROP DISEASE DETECTION REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

DETECTION RESULTS:
- Crop: {disease_info['crop']}
- Disease: {disease_info['disease']}
- Confidence: {confidence:.2f}%
- Severity: {disease_info['severity']}

TREATMENT:
{chr(10).join(['- ' + t for t in disease_info['treatment']])}

PREVENTION:
{chr(10).join(['- ' + p for p in disease_info['prevention']])}

Note: Consult local agricultural experts for confirmation.
            """
            st.download_button(
                label="Download Report as Text",
                data=report,
                file_name=f"disease_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    else:
        # Welcome screen
        st.info("👆 Upload a leaf image to start disease detection")
        
        st.markdown("### 📊 System Capabilities")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", "96.2%")
        with col2:
            st.metric("Diseases", "38")
        with col3:
            st.metric("Crops", "14+")
        with col4:
            st.metric("Speed", "< 3 sec")
        
        st.markdown("---")
        st.markdown("### 📸 Tips for Best Results")
        
        tips_col1, tips_col2 = st.columns(2)
        
        with tips_col1:
            st.success("""
            **✅ Do:**
            - Take photo in natural daylight
            - Focus on infected area
            - Use clear, sharp images
            - Capture single leaf if possible
            """)
        
        with tips_col2:
            st.error("""
            **❌ Avoid:**
            - Dark or blurry images
            - Multiple overlapping leaves
            - Extreme close-ups
            - Photos with heavy shadows
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p><strong>Developed for Indian Farmers</strong></p>
            <p>Built with TensorFlow & Streamlit | Trained on PlantVillage Dataset</p>
            <p style="color: #D32F2F; font-weight: bold;">
                ⚠️ IMPORTANT: This is an AI assistant. For severe infections or commercial farming, 
                always consult your local Krishi Vigyan Kendra (KVK) or agricultural extension officer.
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()