# ==============================================================================
# File Name   : M25CSA003_prob2.py
# Author      : [Your Name Here]
# Roll Number : M25CSA003
# Description : Byte Pair Encoding (BPE) Tokenizer from scratch.
#               This script reads a text corpus, learns merge rules based on
#               frequency, and generates a vocabulary of subword units.
# ==============================================================================

import sys
import collections
import re

# ------------------------------------------------------------------------------
# Function: get_vocab_from_corpus
# Description: Reads the file and converts the corpus into a frequency dictionary.
#              Words are split into characters with a special end-of-word 
#              token </w> to mark boundaries.
# Input: file_path (str)
# Returns: dict {(char_tuple): frequency}
# ------------------------------------------------------------------------------
def get_vocab_from_corpus(file_path):
    vocab = collections.defaultdict(int)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Basic pre-tokenization: split by whitespace
                words = line.strip().split()
                for word in words:
                    # Convert word to characters and append end-of-word symbol
                    # Example: "low" -> ('l', 'o', 'w', '</w>')
                    chars = list(word) + ['</w>']
                    vocab[tuple(chars)] += 1
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
        
    return vocab

# ------------------------------------------------------------------------------
# Function: get_pair_stats
# Description: Counts the frequency of all adjacent character pairs in the
#              current vocabulary.
# Input: vocab (dict)
# Returns: dict {(char1, char2): count}
# ------------------------------------------------------------------------------
def get_pair_stats(vocab):
    pairs = collections.defaultdict(int)
    
    for word_tuple, frequency in vocab.items():
        # Iterate through the characters in the word tuple
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i+1])
            # Add the word's frequency to the pair count
            pairs[pair] += frequency
            
    return pairs

# ------------------------------------------------------------------------------
# Function: merge_vocab
# Description: Merges the most frequent pair in the vocabulary.
#              Replaces instances of (A, B) with (AB).
# Input: pair_to_merge (tuple), vocab (dict)
# Returns: new_vocab (dict)
# ------------------------------------------------------------------------------
def merge_vocab(pair_to_merge, vocab):
    new_vocab = {}
    
    bigram = pair_to_merge # e.g. ('e', 's')
    combined = "".join(bigram) # e.g. "es"
    
    for word_tuple, frequency in vocab.items():
        new_word_tuple = []
        i = 0
        
        # Scan through the word tuple to find the pair
        while i < len(word_tuple):
            # Check if current and next item match the pair we want to merge
            if i < len(word_tuple) - 1 and word_tuple[i] == bigram[0] and word_tuple[i+1] == bigram[1]:
                # Merge them
                new_word_tuple.append(combined)
                i += 2 # Skip the next character since it's merged
            else:
                # Keep original character
                new_word_tuple.append(word_tuple[i])
                i += 1
        
        # Update the dictionary with the new tuple structure
        new_vocab[tuple(new_word_tuple)] = frequency
        
    return new_vocab

# ------------------------------------------------------------------------------
# Function: extract_final_vocabulary
# Description: Extracts unique tokens from the vocabulary dictionary.
# ------------------------------------------------------------------------------
def extract_final_vocabulary(vocab):
    unique_tokens = set()
    for word_tuple in vocab.keys():
        for token in word_tuple:
            unique_tokens.add(token)
    return sorted(list(unique_tokens))

# ------------------------------------------------------------------------------
# Main Execution Block
# ------------------------------------------------------------------------------
def main():
    # 1. Argument Parsing
    if len(sys.argv) < 2:
        print("Usage: python M25CSA003_prob2.py corpus.txt")
        sys.exit(1)
        
    corpus_file = sys.argv[1]
    
    # 2. Get Input for K (Number of Merges)
    try:
        # Ask user for K as per problem description requiring it to "take number of merges K"
        k_input = input("Enter number of merges (K): ")
        num_merges = int(k_input)
    except ValueError:
        print("Invalid input. Please enter an integer for K.")
        sys.exit(1)

    print(f"\n--- Starting BPE Training on '{corpus_file}' with K={num_merges} ---\n")

    # 3. Initialization
    vocab = get_vocab_from_corpus(corpus_file)
    print(f"Initial Vocabulary Size (Unique Words): {len(vocab)}")
    
    # 4. BPE Training Loop
    for i in range(num_merges):
        # Step A: Get pair frequencies
        pairs = get_pair_stats(vocab)
        
        if not pairs:
            print("No more pairs to merge.")
            break
            
        # Step B: Find the best pair (most frequent)
        # We allow ties to be broken arbitrarily (first one found)
        best_pair = max(pairs, key=pairs.get)
        current_freq = pairs[best_pair]
        
        print(f"Merge {i+1}/{num_merges}: Merging pair {best_pair} (Frequency: {current_freq})")
        
        # Step C: Update vocabulary by merging the best pair
        vocab = merge_vocab(best_pair, vocab)

    # 5. Output Final Vocabulary
    final_tokens = extract_final_vocabulary(vocab)
    
    print("\n" + "="*40)
    print("      Final BPE Vocabulary      ")
    print("="*40)
    print(f"Total Tokens: {len(final_tokens)}")
    print("-" * 40)
    # Printing tokens horizontally for better readability
    print(", ".join(f"'{t}'" for t in final_tokens))
    print("="*40)

if __name__ == "__main__":
    main()