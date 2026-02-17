import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os

def generate_dynamic_comparison():
    print("Loading data for evaluation...")
    data = pd.read_csv('data/train.csv')
    data['comment_text'] = data['comment_text'].fillna("unknown")
    
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    X = data['comment_text']
    y = data[labels]
    
    # Use the same split as used in training
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    
    print("Loading models and vectorizers...")
    # Load Naive Bayes
    nb_model = joblib.load('models/nb_model.joblib')
    nb_vectorizer = joblib.load('models/nb_vectorizer.joblib')
    
    # Load Logistic Regression
    lr_model = joblib.load('models/baseline_model.joblib')
    lr_vectorizer = joblib.load('models/baseline_vectorizer.joblib')
    
    print("Evaluating models on validation set...")
    
    # Naive Bayes scores
    X_val_nb = nb_vectorizer.transform(X_val)
    nb_probs = nb_model.predict_proba(X_val_nb)
    nb_scores = []
    for i, label in enumerate(labels):
        score = roc_auc_score(y_val[label], nb_probs[:, i])
        nb_scores.append(score)
        
    # Logistic Regression scores
    X_val_lr = lr_vectorizer.transform(X_val)
    lr_probs = lr_model.predict_proba(X_val_lr)
    lr_scores = []
    for i, label in enumerate(labels):
        score = roc_auc_score(y_val[label], lr_probs[:, i])
        lr_scores.append(score)

    print("\n--- Evaluation Results ---")
    print(f"{'Label':<15} | {'Naive Bayes':<12} | {'LogReg':<12}")
    print("-" * 45)
    for i, label in enumerate(labels):
        print(f"{label:<15} | {nb_scores[i]:.4f}       | {lr_scores[i]:.4f}")
    
    print(f"\nMean Naive Bayes AUC: {np.mean(nb_scores):.4f}")
    print(f"Mean LogReg AUC: {np.mean(lr_scores):.4f}")

    print("\nGenerating comparison chart...")
    display_labels = [l.replace('_', ' ').title() for l in labels]
    x = np.arange(len(display_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, nb_scores, width, label='Naive Bayes', color='skyblue')
    rects2 = ax.bar(x + width/2, lr_scores, width, label='Logistic Regression', color='salmon')

    ax.set_ylabel('ROC-AUC Score')
    ax.set_title('Model Performance Comparison by Category (Actual Data)')
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.set_ylim(0.90, 1.0)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.3f', fontsize=8)
    ax.bar_label(rects2, padding=3, fmt='%.3f', fontsize=8)

    fig.tight_layout()
    plt.savefig('model_comparison_chart.png')
    print("Model comparison chart updated: model_comparison_chart.png")

if __name__ == "__main__":
    generate_dynamic_comparison()
