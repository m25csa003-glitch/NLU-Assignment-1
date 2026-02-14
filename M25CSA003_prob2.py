# ==============================================================================
# File Name   : M25CSA003_prob2.py
# Author      : Akshat Jain
# Roll Number : M25CSA003
# Description : Byte Pair Encoding (BPE) Tokenizer
# Usage       : python M25CSA003_prob2.py <k> <corpus_file>
# ==============================================================================

import sys
import collections
import re

def get_stats(ids, counts):
    """Pair frequencies calculate karta hai."""
    pair_counts = collections.defaultdict(int)
    for word_ids, freq in counts.items():
        for i in range(len(word_ids) - 1):
            pair = (word_ids[i], word_ids[i+1])
            pair_counts[pair] += freq
    return pair_counts

def merge(ids, pair, idx):
    """Sabse zyada aane wale pair ko naye ID se replace karta hai."""
    new_ids = {}
    for word_ids, freq in ids.items():
        new_word_ids = []
        i = 0
        while i < len(word_ids):
            if i < len(word_ids) - 1 and (word_ids[i], word_ids[i+1]) == pair:
                new_word_ids.append(idx)
                i += 2
            else:
                new_word_ids.append(word_ids[i])
                i += 1
        new_ids[tuple(new_word_ids)] = freq
    return new_ids

def main():
    # Command line arguments check (python script.py k corpus.txt)
    if len(sys.argv) != 3:
        print("Usage: python M25CSA003_prob2.py <num_merges_k> <corpus_file>")
        sys.exit(1)

    try:
        k = int(sys.argv[1]) # Pehla argument: k
        corpus_path = sys.argv[2] # Dusra argument: corpus.txt
    except ValueError:
        print("Error: 'k' must be an integer.")
        sys.exit(1)

    # 1. Load Corpus
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: File '{corpus_path}' not found.")
        sys.exit(1)

    # 2. Initialize Vocabulary (Characters + Frequency)
    words = text.split()
    # Word ko characters mein split karke tuple banana
    counts = collections.defaultdict(int)
    for word in words:
        # Preprocessing: character split
        ids = tuple(list(word))
        counts[ids] += 1

    print(f"Initial Vocabulary Size (Unique words): {len(counts)}")
    print(f"Performing {k} merges...\n")

    # 3. BPE Training (k merges)
    merges = {} # (p1, p2) -> new_id
    current_ids = counts

    for i in range(k):
        stats = get_stats(current_ids, counts)
        if not stats:
            break
        
        # Best pair find karo
        best_pair = max(stats, key=stats.get)
        new_id = f"{best_pair[0]}{best_pair[1]}" # String representation for clarity
        
        current_ids = merge(current_ids, best_pair, new_id)
        merges[best_pair] = new_id
        
        print(f"Merge {i+1}: {best_pair} -> {new_id} (Occurrences: {stats[best_pair]})")

    # 4. Final Output
    print("\n--- Final BPE Vocabulary (Sample) ---")
    unique_tokens = set()
    for word_ids in current_ids.keys():
        for token in word_ids:
            unique_tokens.add(token)
    
    print(f"Number of unique tokens after {k} merges: {len(unique_tokens)}")
    print("Sample Tokens:", list(unique_tokens)[:20])

if __name__ == "__main__":
    main()