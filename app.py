import streamlit as st
import pandas as pd
import numpy as np
import joblib

import os

# Set page title and favicon
st.set_page_config(page_title="Toxic Comment Classifier", page_icon="🚫", layout="wide")

# Custom CSS to reduce top margin
st.markdown("""
    <style>
           .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
                padding-left: 5rem;
                padding-right: 5rem;
            }
    </style>
    """, unsafe_allow_html=True)


# Labels
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

@st.cache_resource
def load_models():
    # Load Logistic Regression
    lr_model = joblib.load('models/baseline_model.joblib')
    lr_vectorizer = joblib.load('models/baseline_vectorizer.joblib')
    
    # Load Naive Bayes
    nb_model = joblib.load('models/nb_model.joblib')
    nb_vectorizer = joblib.load('models/nb_vectorizer.joblib')
    
    return lr_model, lr_vectorizer, nb_model, nb_vectorizer

def predict_sklearn(text, model, vectorizer):
    vec_text = vectorizer.transform([text])
    probs = model.predict_proba(vec_text)[0]
    return dict(zip(labels, probs))



# Load models
with st.spinner("Loading models..."):
    lr_model, lr_vectorizer, nb_model, nb_vectorizer = load_models()

# User Input & Results Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.title("🚫 Toxic Comment Classifier")
    st.markdown("Enter a comment below to evaluate toxicity.")
    st.write("---")
    # st.subheader("⌨️ Input")
    user_text = st.text_area("Enter a comment to classify:", height=200)
    selected_model = st.selectbox("Select Model", ["Naive Bayes (TF-IDF)", "Logistic Regression (TF-IDF)"])
    classify_btn = st.button("Classify ✨", use_container_width=True)

with col2:
    st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
    st.subheader("📊 Results")
    if classify_btn:
        if user_text.strip() == "":
            st.warning("Please enter a comment.")
        else:
            results = None
            if "Naive Bayes" in selected_model:
                results = predict_sklearn(user_text, nb_model, nb_vectorizer)
            else:
                results = predict_sklearn(user_text, lr_model, lr_vectorizer)
            
            if results:
                st.write(f"**Model**: {selected_model}")
                
                # Metrics in a grid
                metric_cols = st.columns(2)
                for i, (cat, prob) in enumerate(results.items()):
                    with metric_cols[i % 2]:
                        threshold = 0.5
                        is_toxic = prob >= threshold
                        st.metric(
                            label=cat.replace('_', ' ').title(), 
                            value=f"{prob:.1%}", 
                            delta="TOXIC" if is_toxic else None, 
                            delta_color="inverse" if is_toxic else "normal"
                        )

                st.write("---")
                # Progress bars
                for cat, prob in results.items():
                    st.write(f"**{cat.replace('_', ' ').title()}** ({prob:.1%})")
                    st.progress(prob)
    else:
        st.info("Enter a comment and click 'Classify' to see the results.")
