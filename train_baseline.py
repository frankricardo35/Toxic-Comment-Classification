import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import os

def train_baseline():
    print("Loading data...")
    train = pd.read_csv('data/train.csv')
    
    # Fill missing values
    train['comment_text'] = train['comment_text'].fillna("unknown")
    
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    X = train['comment_text']
    y = train[labels]
    
    # Split for local validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(1, 1),
        max_features=10000
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    print("Training Logistic Regression models...")
    classifier = OneVsRestClassifier(LogisticRegression(C=1.0, solver='sag', max_iter=1000))
    classifier.fit(X_train_vec, y_train)
    
    print("Evaluating...")
    val_preds = classifier.predict_proba(X_val_vec)
    
    auc_scores = []
    for i, label in enumerate(labels):
        score = roc_auc_score(y_val[label], val_preds[:, i])
        auc_scores.append(score)
        print(f"ROC AUC for {label}: {score:.4f}")
    
    mean_auc = np.mean(auc_scores)
    print(f"Mean ROC AUC: {mean_auc:.4f}")
    
    # Save the model and vectorizer
    print("Saving model...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(classifier, 'models/baseline_model.joblib')
    joblib.dump(vectorizer, 'models/baseline_vectorizer.joblib')
    print("Baseline model saved to models/baseline_model.joblib")

if __name__ == "__main__":
    train_baseline()
