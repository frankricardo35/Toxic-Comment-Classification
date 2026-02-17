import joblib
import numpy as np

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def test_predictions():
    test_text = "This is a wonderful and helpful comment!"
    toxic_text = "I hate you and I will hurt you!"

    texts = [test_text, toxic_text]

    print("--- Testing Naive Bayes Model ---")
    nb_model = joblib.load('models/nb_model.joblib')
    nb_vectorizer = joblib.load('models/nb_vectorizer.joblib')
    for text in texts:
        vec = nb_vectorizer.transform([text])
        probs = nb_model.predict_proba(vec)[0]
        print(f"Text: '{text}'")
        for i, l in enumerate(labels):
            print(f"  {l}: {probs[i]:.4f}")

    print("\n--- Testing Logistic Regression Model ---")
    lr_model = joblib.load('models/baseline_model.joblib')
    lr_vectorizer = joblib.load('models/baseline_vectorizer.joblib')
    for text in texts:
        vec = lr_vectorizer.transform([text])
        probs = lr_model.predict_proba(vec)[0]
        print(f"Text: '{text}'")
        for i, l in enumerate(labels):
            print(f"  {l}: {probs[i]:.4f}")

if __name__ == "__main__":
    test_predictions()
