import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda():
    print("Loading data...")
    train = pd.read_csv('data/train.csv')
    
    # 1. Basic Stats
    print("\nDataset Shape:", train.shape)
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # 2. Label Distribution
    print("\n--- Label Distribution ---")
    counts = train[labels].sum()
    print(counts)
    
    # Plotting label distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    plt.title('Distribution of Toxic Comment Labels')
    plt.ylabel('Number of Comments')
    plt.xlabel('Category')
    plt.savefig('label_distribution.png')
    print("Saved label_distribution.png")

    # 3. Clean vs Toxic Comments
    train['clean'] = (train[labels].sum(axis=1) == 0).astype(int)
    clean_counts = train['clean'].value_counts()
    print("\nClean vs Toxic:")
    print(f"  Clean: {clean_counts[1]} ({clean_counts[1]/len(train):.2%})")
    print(f"  Toxic (any): {clean_counts[0]} ({clean_counts[0]/len(train):.2%})")

    # 4. Correlation between labels
    plt.figure(figsize=(10, 8))
    sns.heatmap(train[labels].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Toxicity Labels')
    plt.savefig('label_correlation.png')
    print("Saved label_correlation.png")

    # 5. Text Length Analysis
    train['char_length'] = train['comment_text'].apply(lambda x: len(str(x)))
    print("\nText Length Statistics:")
    print(train['char_length'].describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(train['char_length'], bins=50, kde=True, color='blue')
    plt.title('Distribution of Comment Character Length')
    plt.xlabel('Character Length')
    plt.ylabel('Frequency')
    plt.savefig('text_length_distribution.png')
    print("Saved text_length_distribution.png")

    print("\nEDA Completed successfully.")

if __name__ == "__main__":
    os.makedirs('eda_results', exist_ok=True)
    # Move results to a folder if desired, but for now keeping at root for easy access
    perform_eda()
