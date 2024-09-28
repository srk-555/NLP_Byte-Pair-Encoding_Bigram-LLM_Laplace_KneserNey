#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imorting necessary libraries
import pandas as pd
import numpy as np 
import warnings
warnings.filterwarnings('ignore')

from collections import defaultdict
from typing import List

# Our Tokenizer Class
class Tokenizer:
    # Class Constructor to declare and initialize variables
    def __init__(self):
        self.vocab = set()  # Vocabulary, unique tokens in the corpus
        self.vocab.add("$")
        self.frequency_corpus = defaultdict(int)  # Unigram counts, frequency of each unique token
        self.merge_rules = []    # All the merge rules learned and to be appended in order as they are learned 

    # a function that learns the vocabulary
    def learn_vocabulary(self, corpus: List[str], num_merges: int):
        # Initialize vocabulary with characters and their frequencies
        
        # Construct a frequency corpus i.e. Unigram counts of unique tokens
        for sentence in corpus:
            for word in sentence.split():
                modified_word = ""
                for char in word:
                    self.vocab.add(char)
                    modified_word = modified_word + char + "-"
                modified_word = modified_word + "$"
                self.frequency_corpus[modified_word] += 1

        # Learn the merge rules
        for merge in range(num_merges):
            pair_frequency = defaultdict(int)     # This stores the character pair frequency for each merge rule iteration 
            
            # Generating pair frequencies
            for word in self.frequency_corpus:
                symbols = word.split("-")
                for i in range(len(symbols) - 1):
                    pair_frequency[(symbols[i], symbols[i + 1])] += 1

            if not pair_frequency:
                break

            # Get the character pair frequency with maximum count
            best_pair = max(pair_frequency, key=pair_frequency.get)
            self.vocab.add(best_pair[0]+best_pair[1])
            
            # Add that pair with maximum frequency in the merge rules if it doesn't exist
            if best_pair not in self.merge_rules:
                self.merge_rules.append(best_pair)

            # Update the frequency corpus for next iteration
            new_frequency_corpus = defaultdict(int)
            for word in self.frequency_corpus:
                new_word = word.replace(best_pair[0]+"-"+best_pair[1], best_pair[0]+best_pair[1])
                new_frequency_corpus[new_word] = self.frequency_corpus[word]
            self.frequency_corpus = new_frequency_corpus
        
        ### WRITING MERGE RULES TO TEXT FILE
        # Clear past contents of the file
        try:
            with open("merge_rules.txt", "w") as file:
                pass
        except Exception as e:
            print(f"Error clearing file: {e}")
            
        # Write all merge_rules to merge_rules.txt for submittables
        for rule in self.merge_rules:
            r = ",".join(rule)
            try:
                with open("merge_rules.txt", "a") as file:
                    file.write(r + "\n")
            except Exception as e:
                print(f"Error writing merge rule to merge_rules.txt: {e}")
            
        
        ### WRITING VOCABULARY TO TEXT FILE
        # Clear past contents of the file
        try:
            with open("vocab.txt", "w") as file:
                pass
        except Exception as e:
            print(f"Error clearing file: {e}")
            
        # Write all tokens found to tokens.txt for submittables
        tokens = list(self.vocab)
        tokens = sorted(tokens, key=lambda x: (len(x), x))
        for token in tokens:
            try:
                with open("vocab.txt", "a") as file:
                    file.write(token + "\n")
            except Exception as e:
                print(f"Error writing token to vocab.txt: {e}")
        
        

