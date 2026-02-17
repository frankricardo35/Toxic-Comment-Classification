# 🚫 Toxic Comment Classification Project

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

## 📝 Project Summary
This project provides a robust solution for identifying toxic behavior in online comments. By leveraging natural language processing (NLP) and machine learning, the system classifies text into six distinct categories of toxicity. It serves as a comparative study between a probabilistic baseline (Naive Bayes) and a discriminative baseline (Logistic Regression), both optimized using TF-IDF feature extraction. The final app provides an intuitive, responsive interface for real-time comment moderation.

## 📊 Dataset

The dataset used is from the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data) on Kaggle.

### Data Requirements
To run this project, download the following files from Kaggle and place them in a `data/` directory:
- `train.csv` (Training data with labels)
- `test.csv` (Test data for predictions)
- `test_labels.csv` (Labels for the test set)

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python installed, then install the required dependencies using the provided requirements file:
```bash
pip install -r requirements.txt
```

### 2. Execution Sequence
Follow these steps in order to reproduce the analysis and train the models:

#### Step A: Exploratory Data Analysis (EDA)
Generate data insights and visualizations:
```bash
python3 perform_eda.py
```

#### Step B: Training the Models
Train both models and save them to the `models/` directory:
```bash
python3 train_baseline.py
python3 train_nb.py
```

#### Step C: Evaluation & Comparison
Compare model performance and generate category-wise charts:
```bash
python3 compare_models.py
```

## 🌐 Running the App

Launch the interactive Streamlit interface to test custom comments:
```bash
streamlit run app.py
```

## 📈 Results
- **Logistic Regression**: ~0.9793 Mean ROC-AUC
- **Naive Bayes**: ~0.9658 Mean ROC-AUC
