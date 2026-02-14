# ==============================================================================
# File Name   : classifier.py
# Author      : Akshat Jain (M25CSA003)
# Description : Sports vs Politics Classifier
#               Compare 3 Models: Naive Bayes, Logistic Regression, Random Forest
#               Includes Interactive Prediction Mode.
# ==============================================================================

import pandas as pd
import numpy as np
import sys

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------------------------------------------
# 1. Configuration & Setup
# ------------------------------------------------------------------------------
CSV_FILE = 'bbc_news_text_complexity_summarization.csv'
TARGET_LABELS = ['sport', 'politics']

def load_and_preprocess_data(filepath):
    """
    Load CSV, filter for specific labels, and clean text.
    """
    print(f"Loading dataset from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found. Make sure it is in the same folder.")
        sys.exit(1)

    # Filter only Sports and Politics
    df_filtered = df[df['labels'].isin(TARGET_LABELS)].copy()
    
    # Drop rows with missing values
    df_filtered = df_filtered[['text', 'labels']].dropna()

    # Basic Cleaning: Lowercase and remove newlines
    df_filtered['text'] = df_filtered['text'].str.replace('\n', ' ').str.lower().str.strip()
    
    print(f"Data loaded successfully! Total samples: {len(df_filtered)}")
    print(f"Sports: {len(df_filtered[df_filtered['labels']=='sport'])}")
    print(f"Politics: {len(df_filtered[df_filtered['labels']=='politics'])}")
    
    return df_filtered

# ------------------------------------------------------------------------------
# 2. Model Training & Evaluation
# ------------------------------------------------------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Train 3 different models and print their performance.
    Returns the best model for interactive use.
    """
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    best_model = None
    best_accuracy = 0

    print("\n" + "="*60)
    print(f"{'Model Name':<25} | {'Accuracy':<10}")
    print("="*60)

    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        print(f"{name:<25} | {acc*100:.2f}%")
        
        # Keep track of best model (Logistic Regression usually wins here)
        if acc >= best_accuracy:
            best_accuracy = acc
            best_model = model

    print("="*60)
    return best_model

# ------------------------------------------------------------------------------
# 3. Main Execution
# ------------------------------------------------------------------------------
def main():
    # Step 1: Prepare Data
    df = load_and_preprocess_data(CSV_FILE)
    
    X = df['text']
    y = df['labels']

    # Step 2: Vectorization (TF-IDF)
    print("\nExtracting features using TF-IDF...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    
    # Fit on all data to ensure vocabulary consistency
    X_vectorized = tfidf.fit_transform(X)
    
    # Step 3: Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 4: Run Models
    best_model = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Step 5: Interactive Mode
    print("\n--- Interactive Mode ---")
    print("Type a news headline to classify (or type 'exit' to quit).")
    
    while True:
        try:
            user_input = input("\nEnter text: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
            
            # Preprocess user input exactly like training data
            cleaned_input = user_input.lower().replace('\n', ' ').strip()
            input_vector = tfidf.transform([cleaned_input])
            
            # Predict
            prediction = best_model.predict(input_vector)[0]
            probability = best_model.predict_proba(input_vector).max() * 100
            
            # Show Result with Confidence
            icon = "" if prediction == "sport" else ""
            print(f"Prediction: {icon} {prediction.upper()} (Confidence: {probability:.2f}%)")
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()