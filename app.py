# Sentiment Analysis Dashboard
# A complete project demonstrating ML skills with sentiment analysis, data visualization,
# and a simple web interface

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import string
import pickle
import streamlit as st
import time
from wordcloud import WordCloud
import joblib

# Ensure required NLTK resources are downloaded
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

# Text preprocessing functions
def clean_text(text):
    """Basic text cleaning: lowercase, remove punctuation and numbers"""
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def remove_stopwords(text):
    """Remove common stopwords"""
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    """Lemmatize text to reduce words to their base form"""
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def preprocess_text(text):
    """Apply all preprocessing steps"""
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

# Load and prepare dataset
def load_data():
    """
    Load dataset from a local file if present, otherwise download from URL
    Using Twitter Sentiment Analysis Dataset as an example
    """
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Local path
    data_path = os.path.join(data_dir, "sentiment_data.csv")
    
    # Check if data exists locally
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    
    # Otherwise, use a sample dataset or download one
    # For this example, we'll create a small sample dataset
    st.warning("Dataset not found locally. Creating a sample dataset...")
    
    # Sample dataset with some positive and negative reviews
    sample_data = {
        'text': [
            "I love this product, it's amazing!",
            "The service was excellent and I would recommend it to everyone.",
            "This is the worst experience I've ever had.",
            "Don't waste your money on this garbage.",
            "The staff was friendly and helpful.",
            "Great value for money, would buy again.",
            "Terrible customer service, never coming back.",
            "I'm so disappointed with the quality.",
            "Absolutely fantastic product, exceeded my expectations!",
            "This company doesn't care about its customers."
        ],
        'sentiment': [1, 1, 0, 0, 1, 1, 0, 0, 1, 0]  # 1 for positive, 0 for negative
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(data_path, index=False)
    return df

# Train models
def train_models(X, y):
    """Train multiple ML models for sentiment analysis"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create text processing and model pipeline
    models = {
        'Naive Bayes': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', MultinomialNB())
        ]),
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', LogisticRegression(max_iter=1000))
        ]),
        'Random Forest': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', RandomForestClassifier(n_estimators=100))
        ])
    }
    
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Test model
        y_pred = model.predict(X_test)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        # Save the model
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, f'models/{name.replace(" ", "_").lower()}_model.joblib')
    
    return results

# Visualization functions
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt

def plot_class_distribution(y):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment (0=Negative, 1=Positive)')
    plt.ylabel('Count')
    return plt

def generate_wordcloud(texts, sentiments, sentiment_value):
    """Generate wordcloud for positive or negative sentiments"""
    # Filter texts by sentiment
    filtered_texts = [text for text, sent in zip(texts, sentiments) if sent == sentiment_value]
    
    if not filtered_texts:
        return None
    
    # Combine all texts
    combined_text = ' '.join(filtered_texts)
    
    # Generate wordcloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=100
    ).generate(combined_text)
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    sentiment_label = 'Positive' if sentiment_value == 1 else 'Negative'
    plt.title(f'Word Cloud for {sentiment_label} Sentiments')
    return plt

# Streamlit app functions
def build_dashboard():
    st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
    
    st.title("Sentiment Analysis Dashboard")
    st.write("""
    This dashboard demonstrates advanced NLP and machine learning techniques for sentiment analysis.
    Upload your own dataset or use the provided sample to analyze text sentiment.
    """)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Exploration", "Model Training & Evaluation", "Prediction"])
    
    # Download necessary NLTK resources
    download_nltk_resources()
    
    # Load data
    df = load_data()
    
    # Preprocess text if not already done
    if 'processed_text' not in df.columns:
        with st.spinner("Preprocessing text data..."):
            df['processed_text'] = df['text'].apply(preprocess_text)
    
    if page == "Data Exploration":
        show_data_exploration(df)
    
    elif page == "Model Training & Evaluation":
        show_model_training(df)
    
    elif page == "Prediction":
        show_prediction_page(df)

def show_data_exploration(df):
    st.header("Data Exploration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Data Sample")
        st.dataframe(df.head())
        
        st.subheader("Data Distribution")
        fig = plot_class_distribution(df['sentiment'])
        st.pyplot(fig)
    
    with col2:
        st.subheader("Word Clouds by Sentiment")
        
        # Positive sentiment wordcloud
        fig_pos = generate_wordcloud(df['processed_text'], df['sentiment'], 1)
        if fig_pos:
            st.pyplot(fig_pos)
        
        # Negative sentiment wordcloud
        fig_neg = generate_wordcloud(df['processed_text'], df['sentiment'], 0)
        if fig_neg:
            st.pyplot(fig_neg)
    
    st.subheader("Text Preprocessing Example")
    
    example_idx = st.selectbox("Select example text:", range(len(df)))
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Text:**")
        st.write(df['text'].iloc[example_idx])
    
    with col2:
        st.markdown("**Processed Text:**")
        st.write(df['processed_text'].iloc[example_idx])

def show_model_training(df):
    st.header("Model Training & Evaluation")
    
    if st.button("Train Models"):
        with st.spinner("Training models. This may take a few minutes..."):
            results = train_models(df['processed_text'], df['sentiment'])
            
            # Save results to session state
            st.session_state.model_results = results
            st.success("Models trained successfully!")
    
    # If models are trained, show results
    if 'model_results' in st.session_state:
        results = st.session_state.model_results
        
        # Model performance comparison
        st.subheader("Model Performance Comparison")
        
        # Create dataframe for model comparison
        model_accuracies = {name: result['accuracy'] for name, result in results.items()}
        accuracy_df = pd.DataFrame.from_dict(model_accuracies, orient='index', columns=['Accuracy'])
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.dataframe(accuracy_df)
        
        with col2:
            # Plot model accuracies
            fig, ax = plt.subplots(figsize=(8, 4))
            accuracy_df.plot(kind='bar', ax=ax)
            ax.set_ylabel('Accuracy')
            ax.set_title('Model Accuracy Comparison')
            st.pyplot(fig)
        
        # Detailed model evaluation
        st.subheader("Detailed Model Evaluation")
        
        selected_model = st.selectbox("Select model:", list(results.keys()))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion matrix
            st.markdown(f"**Confusion Matrix - {selected_model}**")
            fig = plot_confusion_matrix(results[selected_model]['confusion_matrix'], selected_model)
            st.pyplot(fig)
        
        with col2:
            # Classification report
            st.markdown(f"**Classification Report - {selected_model}**")
            report = results[selected_model]['report']
            # Convert classification report to dataframe for better display
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

def show_prediction_page(df):
    st.header("Sentiment Prediction")
    
    # Check if models are available
    models_dir = "models"
    models_available = os.path.exists(models_dir) and len(os.listdir(models_dir)) > 0
    
    if not models_available:
        st.warning("No trained models found. Please go to the 'Model Training & Evaluation' page to train models first.")
        return
    
    # Load available models
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    model_names = [f.replace('_model.joblib', '').replace('_', ' ').title() for f in model_files]
    
    # Model selection
    selected_model_name = st.selectbox("Select model for prediction:", model_names)
    model_file = f"models/{selected_model_name.lower().replace(' ', '_')}_model.joblib"
    
    model = joblib.load(model_file)
    
    # Text input
    st.subheader("Enter text for sentiment analysis:")
    user_text = st.text_area("", "Type your text here...")
    
    if st.button("Analyze Sentiment"):
        if user_text:
            # Preprocess text
            processed_text = preprocess_text(user_text)
            
            # Make prediction
            prediction = model.predict([processed_text])[0]
            probability = model.predict_proba([processed_text])[0]
            
            # Display result
            sentiment = "Positive" if prediction == 1 else "Negative"
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display prediction
                if sentiment == "Positive":
                    st.success(f"Sentiment: {sentiment}")
                else:
                    st.error(f"Sentiment: {sentiment}")
                
                st.write(f"Confidence: {probability.max():.2f}")
            
            with col2:
                # Visualization
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(['Negative', 'Positive'], probability, color=['#ff9999', '#66b3ff'])
                ax.set_ylim(0, 1)
                ax.set_title('Prediction Probability')
                st.pyplot(fig)
            
            # Show preprocessing steps
            with st.expander("View text preprocessing steps"):
                st.write("Original text:")
                st.write(user_text)
                
                st.write("After cleaning (lowercase, removing punctuation, numbers):")
                st.write(clean_text(user_text))
                
                st.write("After removing stopwords:")
                st.write(remove_stopwords(clean_text(user_text)))
                
                st.write("After lemmatization (final processed text):")
                st.write(processed_text)

if __name__ == "__main__":
    build_dashboard()