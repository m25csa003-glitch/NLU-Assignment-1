# ==============================================================================
# File Name   : M25CSA003_prob3.py
# Author      : Akshat Jain
# Roll Number : M25CSA003
# Description : Naive Bayes Sentiment Classifier (Optimized)
#               - Implements custom preprocessing (Stop words, Negation handling)
#               - Trains a Naive Bayes model with Laplace Smoothing
#               - Provides interactive prediction
# ==============================================================================

import math
import random
import sys
from collections import defaultdict

# ------------------------------------------------------------------------------
# Preprocessing Configuration
# ------------------------------------------------------------------------------
STOP_WORDS = {
    'the', 'is', 'and', 'a', 'an', 'of', 'to', 'in', 'it', 'this', 'that', 
    'was', 'for', 'with', 'as', 'at', 'by', 'on', 'be', 'are', 'i', 'my', 
    'have', 'had', 'has', 'very', 'am', 'so'
}

PUNCTUATION = '.,!?;:"()-'

# ------------------------------------------------------------------------------
# Function: preprocess_text
# Description: Cleans text by removing punctuation, stop words, and handling
#              negation (e.g., "not good" -> "not_good").
# Input: raw_text (str)
# Returns: list of strings (tokens)
# ------------------------------------------------------------------------------
def preprocess_text(raw_text):
    # 1. Lowercase and remove punctuation
    clean_text = raw_text.lower()
    for char in PUNCTUATION:
        clean_text = clean_text.replace(char, ' ')
    
    words = clean_text.split()
    tokens = []
    
    skip_next = False
    
    for i in range(len(words)):
        if skip_next:
            skip_next = False
            continue
            
        word = words[i]
        
        # 2. Check for negation terms
        if word in ["not", "no", "never", "n't", "dont", "didnt", "wont"]:
            # If there is a next word, combine them (e.g., "not_good")
            if i + 1 < len(words):
                negated_word = "not_" + words[i+1]
                tokens.append(negated_word)
                skip_next = True # Skip the next word since we merged it
            else:
                tokens.append(word) # Keep 'not' if it's the last word
        
        # 3. Filter out Stop Words (only if not part of negation)
        elif word not in STOP_WORDS:
            tokens.append(word)
            
    return tokens

# ------------------------------------------------------------------------------
# Function: load_data
# Description: Reads file and applies preprocessing.
# ------------------------------------------------------------------------------
def load_data(filename, label):
    data = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = preprocess_text(line)
                    if tokens: # Only add if tokens exist after cleaning
                        data.append((tokens, label))
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    return data

# ------------------------------------------------------------------------------
# Class: NaiveBayesClassifier
# ------------------------------------------------------------------------------
class NaiveBayesClassifier:
    def __init__(self):
        self.log_priors = {}        
        self.word_counts = {}       
        self.class_total_words = {} 
        self.vocab = set()          
        self.classes = set()        

    def train(self, training_data):
        class_doc_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_total_words = defaultdict(int)
        
        total_docs = len(training_data)

        for tokens, label in training_data:
            self.classes.add(label)
            class_doc_counts[label] += 1
            
            for word in tokens:
                self.vocab.add(word)
                self.word_counts[label][word] += 1
                self.class_total_words[label] += 1

        # Calculate Log Priors
        for label in self.classes:
            prob = class_doc_counts[label] / total_docs
            self.log_priors[label] = math.log(prob)

    def predict(self, sentence_tokens):
        scores = {}
        vocab_size = len(self.vocab)

        for label in self.classes:
            scores[label] = self.log_priors[label]

            for word in sentence_tokens:
                # Laplace Smoothing
                # Unknown words get count 0
                numerator = self.word_counts[label][word] + 1
                denominator = self.class_total_words[label] + vocab_size
                
                scores[label] += math.log(numerator / denominator)

        return max(scores, key=scores.get)

# ------------------------------------------------------------------------------
# Function: evaluate
# ------------------------------------------------------------------------------
def evaluate(model, validation_data):
    correct = 0
    for tokens, label in validation_data:
        prediction = model.predict(tokens)
        if prediction == label:
            correct += 1
    return correct / len(validation_data) if validation_data else 0

# ------------------------------------------------------------------------------
# Main Execution Block
# ------------------------------------------------------------------------------
def main():
    print("\n--- Naive Bayes Sentiment Classifier (Optimized) ---")
    
    # 1. Load Data
    print("Loading and preprocessing data...")
    pos_data = load_data('pos.txt', 'POSITIVE')
    neg_data = load_data('neg.txt', 'NEGATIVE')
    
    all_data = pos_data + neg_data
    random.seed(42) # Set seed for reproducibility
    random.shuffle(all_data)
    
    # 2. Split Data (80% Train, 20% Validation)
    split_index = int(0.8 * len(all_data))
    train_data = all_data[:split_index]
    val_data = all_data[split_index:]
    
    print(f"Total Sentences : {len(all_data)}")
    print(f"Training Set    : {len(train_data)}")
    print(f"Validation Set  : {len(val_data)}")
    
    # 3. Train Model
    print("\nTraining model...")
    classifier = NaiveBayesClassifier()
    classifier.train(train_data)
    
    # 4. Evaluate
    accuracy = evaluate(classifier, val_data)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print("-" * 40)

    # 5. Interactive Loop
    print("Enter a sentence to classify (or type 'exit' to quit).")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("Enter sentence: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
            
            # CRITICAL: Apply same preprocessing to user input
            tokens = preprocess_text(user_input)
            
            # Predict
            sentiment = classifier.predict(tokens)
            
            # Debug: Show how the model sees the input (optional)
            # print(f"[Debug] Tokens: {tokens}") 
            
            print(f"Prediction: {sentiment}")
            print("-" * 20)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()