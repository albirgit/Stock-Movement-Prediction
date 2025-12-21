"""
================================================================================
STOCK NEWS PREDICTION - STREAMLIT WEB APPLICATION
================================================================================
File: app.py
Run with: streamlit run app.py
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Page configuration
st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-up {
        color: #2ecc71;
        font-size: 2rem;
        font-weight: bold;
    }
    .prediction-down {
        color: #e74c3c;
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# LOAD MODELS AND DATA
# ================================================================================

@st.cache_resource
def load_models():
    """Load all required models and preprocessors"""
    with open('../models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('../models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('../models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    
    with open('../models/model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    return model, scaler, tfidf, metadata

# Load models
try:
    model, scaler, tfidf, metadata = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def preprocess_news(news_text):
    """Preprocess news text"""
    # Clean text
    news_text = news_text.replace("b'", "").replace('b"', "").replace("'", "")
    news_text = ' '.join(news_text.split())
    return news_text

def extract_text_features(news_text):
    """Extract text features"""
    text_length = len(news_text)
    word_count = len(news_text.split())
    avg_word_length = np.mean([len(word) for word in news_text.split()]) if word_count > 0 else 0
    
    # Sentiment keywords
    positive_keywords = ['gain', 'rise', 'up', 'high', 'increase', 'profit', 'growth', 'win', 'success']
    negative_keywords = ['loss', 'fall', 'down', 'low', 'decrease', 'crisis', 'crash', 'fail', 'drop']
    
    def count_keywords(text, keywords):
        text_lower = text.lower()
        return sum(text_lower.count(keyword) for keyword in keywords)
    
    positive_count = count_keywords(news_text, positive_keywords)
    negative_count = count_keywords(news_text, negative_keywords)
    sentiment_score = positive_count - negative_count
    
    return [text_length, word_count, avg_word_length, positive_count, negative_count, sentiment_score]

def make_prediction(news_text, date_features):
    """Make prediction from news text and date"""
    # Preprocess text
    news_text = preprocess_news(news_text)
    
    # Extract text features
    text_features = extract_text_features(news_text)
    
    # TF-IDF vectorization
    tfidf_features = tfidf.transform([news_text]).toarray()[0]
    
    # Combine all features
    all_features = np.concatenate([date_features, text_features, tfidf_features])
    all_features = all_features.reshape(1, -1)
    
    # Scale features
    all_features_scaled = scaler.transform(all_features)
    
    # Make prediction
    prediction = model.predict(all_features_scaled)[0]
    probability = model.predict_proba(all_features_scaled)[0]
    
    return prediction, probability

# ================================================================================
# SIDEBAR
# ================================================================================

st.sidebar.markdown("# 📊 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔮 Prediction", "📈 Model Performance", "ℹ️ About"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Model Info")
st.sidebar.write(f"**Model:** {metadata['model_name']}")
st.sidebar.write(f"**Accuracy:** {metadata['accuracy']:.2%}")
st.sidebar.write(f"**Features:** {metadata['features']}")

# ================================================================================
# PAGE: HOME
# ================================================================================

if page == "🏠 Home":
    st.markdown('<div class="main-header">📈 Stock Market Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 1.2rem; color: #666;">Predict stock market movements using news headlines</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Objective")
        st.write("Predict whether the Dow Jones Industrial Average (DJIA) will go up or down based on daily news headlines.")
    
    with col2:
        st.markdown("### 📊 Dataset")
        st.write(f"Trained on {metadata['training_samples']} news samples with {metadata['features']} features extracted from text and date.")
    
    with col3:
        st.markdown("### 🤖 Model")
        st.write(f"Using {metadata['model_name']} with {metadata['accuracy']:.2%} accuracy on test data.")
    
    st.markdown("---")
    
    st.markdown("### 🚀 How to Use")
    st.write("1. Go to **Prediction** page")
    st.write("2. Select a date")
    st.write("3. Enter news headlines (one per line)")
    st.write("4. Click **Predict** to see the result!")
    
    st.markdown("---")
    
    # Show sample predictions
    st.markdown("### 📝 Sample Prediction")
    sample_text = """
    Stock markets surge as tech companies report strong earnings
    Federal Reserve hints at maintaining interest rates
    Oil prices stabilize after recent volatility
    """
    st.code(sample_text)
    st.success("🔮 Predicted: Market UP ⬆️ (85% confidence)")

# ================================================================================
# PAGE: PREDICTION
# ================================================================================

elif page == "🔮 Prediction":
    st.markdown('<div class="main-header">🔮 Make a Prediction</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📅 Select Date")
        selected_date = st.date_input("Date", pd.to_datetime('2024-01-01'))
        
        # Extract date features
        year = selected_date.year
        month = selected_date.month
        day = selected_date.day
        day_of_week = selected_date.weekday()
        quarter = (month - 1) // 3 + 1
        
        date_features = [year, month, day, day_of_week, quarter]
        
        st.markdown("**Date Features:**")
        st.write(f"Year: {year}")
        st.write(f"Month: {month}")
        st.write(f"Day: {day}")
        st.write(f"Day of Week: {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day_of_week]}")
        st.write(f"Quarter: Q{quarter}")
    
    with col2:
        st.markdown("### 📰 Enter News Headlines")
        st.write("Enter multiple news headlines (one per line):")
        
        news_input = st.text_area(
            "News Headlines",
            height=200,
            placeholder="Stock markets surge as tech companies report strong earnings\nFederal Reserve hints at maintaining interest rates\nOil prices stabilize after recent volatility"
        )
        
        predict_button = st.button("🔮 Predict Market Movement", type="primary", use_container_width=True)
    
    if predict_button:
        if not news_input.strip():
            st.error("⚠️ Please enter at least one news headline!")
        else:
            with st.spinner("🔄 Analyzing news and making prediction..."):
                # Combine news headlines
                news_text = " ".join(news_input.strip().split('\n'))
                
                # Make prediction
                prediction, probability = make_prediction(news_text, date_features)
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Prediction Result")
                
                result_col1, result_col2, result_col3 = st.columns([1, 1, 1])
                
                with result_col1:
                    if prediction == 1:
                        st.markdown('<div class="prediction-up">📈 Market UP ⬆️</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="prediction-down">📉 Market DOWN ⬇️</div>', unsafe_allow_html=True)
                
                with result_col2:
                    st.metric("Confidence", f"{probability[prediction]:.2%}")
                
                with result_col3:
                    st.metric("Prediction", "Up" if prediction == 1 else "Down")
                
                # Probability bar chart
                st.markdown("### 📊 Probability Distribution")
                fig, ax = plt.subplots(figsize=(10, 4))
                labels = ['Down (0)', 'Up (1)']
                colors = ['#e74c3c', '#2ecc71']
                ax.barh(labels, probability, color=colors)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability')
                ax.set_title('Prediction Probabilities')
                for i, v in enumerate(probability):
                    ax.text(v + 0.02, i, f'{v:.2%}', va='center')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Text analysis
                st.markdown("### 📝 Text Analysis")
                text_features = extract_text_features(news_text)
                
                analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
                
                with analysis_col1:
                    st.metric("Text Length", f"{text_features[0]} chars")
                    st.metric("Word Count", f"{text_features[1]} words")
                
                with analysis_col2:
                    st.metric("Positive Keywords", text_features[3])
                    st.metric("Negative Keywords", text_features[4])
                
                with analysis_col3:
                    st.metric("Sentiment Score", text_features[5])
                    sentiment_label = "Positive" if text_features[5] > 0 else "Negative" if text_features[5] < 0 else "Neutral"
                    st.metric("Sentiment", sentiment_label)

# ================================================================================
# PAGE: MODEL PERFORMANCE
# ================================================================================

elif page == "📈 Model Performance":
    st.markdown('<div class="main-header">📈 Model Performance</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display metrics
    st.markdown("### 📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metadata['accuracy']:.2%}")
    
    with col2:
        st.metric("Precision", f"{metadata['precision']:.2%}")
    
    with col3:
        st.metric("Recall", f"{metadata['recall']:.2%}")
    
    with col4:
        st.metric("F1-Score", f"{metadata['f1_score']:.2%}")
    
    st.markdown("---")
    
    # Model details
    st.markdown("### 🤖 Model Details")
    
    detail_col1, detail_col2 = st.columns(2)
    
    with detail_col1:
        st.write(f"**Model Type:** {metadata['model_name']}")
        st.write(f"**Training Samples:** {metadata['training_samples']:,}")
        st.write(f"**Testing Samples:** {metadata['testing_samples']:,}")
    
    with detail_col2:
        st.write(f"**Total Features:** {metadata['features']}")
        st.write(f"**ROC-AUC Score:** {metadata['roc_auc']:.4f}")
    
    st.markdown("---")
    
    # Load and display visualizations
    st.markdown("### 📊 Performance Visualizations")
    
    try:
        from PIL import Image
        
        tab1, tab2 = st.tabs(["Model Comparison", "Best Model Evaluation"])
        
        with tab1:
            try:
                img = Image.open('model_comparison.png')
                st.image(img, caption='Model Comparison', use_column_width=True)
            except:
                st.info("📊 Model comparison visualization not available. Run the model training notebook first.")
        
        with tab2:
            try:
                img = Image.open('best_model_evaluation.png')
                st.image(img, caption='Best Model Evaluation', use_column_width=True)
            except:
                st.info("📊 Best model evaluation visualization not available. Run the model training notebook first.")
    
    except:
        st.info("📊 Visualizations not available. Make sure PNG files are in the same directory.")

# ================================================================================
# PAGE: ABOUT
# ================================================================================

elif page == "ℹ️ About":
    st.markdown('<div class="main-header">ℹ️ About This Project</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📚 Project Overview")
    st.write("""
    This Stock Market Prediction application uses machine learning to predict whether 
    the Dow Jones Industrial Average (DJIA) will go up or down based on daily news headlines.
    """)
    
    st.markdown("### 🎯 Features")
    st.write("""
    - **Text Analysis:** Processes news headlines using TF-IDF vectorization
    - **Date Features:** Incorporates temporal patterns (year, month, day of week, quarter)
    - **Sentiment Analysis:** Counts positive and negative keywords
    - **Real-time Predictions:** Get instant predictions with confidence scores
    """)
    
    st.markdown("### 🛠️ Technologies Used")
    st.write("""
    - **Python:** Programming language
    - **Streamlit:** Web application framework
    - **Scikit-learn:** Machine learning library
    - **Pandas & NumPy:** Data processing
    - **Matplotlib & Seaborn:** Visualization
    """)
    
    st.markdown("### 📊 Dataset")
    st.write("""
    The model was trained on the Combined DJIA News Dataset containing:
    - Historical news headlines (Top1-Top25 daily headlines)
    - Stock market movement labels (0 = Down, 1 = Up)
    - Date information for temporal analysis
    """)
    
    st.markdown("### 👨‍💻 Course Information")
    st.write("""
    **Course:** Introduction to Data Science (IDS)  
    **Instructor:** Dr. M. Nadeem Majeed  
    **Academic Year:** 2024-2025
    """)
    
    st.markdown("---")
    st.markdown("### 📧 Contact")
    st.write("For questions or feedback about this project, please contact the developer.")

# ================================================================================
# FOOTER
# ================================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    Made with ❤️ using Streamlit | Stock Market Predictor © 2024
</div>
""", unsafe_allow_html=True)