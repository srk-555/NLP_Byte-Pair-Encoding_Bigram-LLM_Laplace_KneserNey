#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install transformers')

from transformers import pipeline
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True,)
def emotion_scores(sample): 
    emotion=classifier(sample)
    return emotion[0]


import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
import warnings
warnings.filterwarnings('ignore')

class BigramLM:
    def __init__(self):
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.unigram_counts = defaultdict(int)
        self.bigramDictionary = defaultdict(int)
        self.bigramProbabilities = defaultdict(int)
        self.laplaceProbabilities = defaultdict(int)
        self.kneserneyProbabilities = defaultdict(int)
        self.vocab = set()
        self.discount = 0.75               # Kneser Ney Discount Factor
        self.probability_comparison_frame = pd.DataFrame()
        self.probability_comparison_frame_available = pd.DataFrame()
        self.bigram_emotion_dictionary = defaultdict(list)

    # Task2: Subtask1: method to learn the model
    def learn_model(self, dataset):
        for sentence in dataset:
            tokens = sentence
            tokens.append('$')  # Adding End of Sentence marker
            
            # Construct a bigram and unigram counts dictionary for further reference
            for i in range(len(tokens)):
                prev_token = tokens[i - 1]
                current_token = tokens[i]

                self.bigram_counts[prev_token][current_token] += 1
                self.unigram_counts[prev_token] += 1
                self.vocab.add(prev_token)
                self.bigramDictionary[(prev_token, current_token)] += 1

        self.calculate_probability()
        # self.add_emotion_component_to_bigram_probabilities() # Commenting as already created and stored using pickle
        self.import_stored_pickle_file()
        self.build_laplace_and_kneserNey_probabilities()

    
    # Rank top 50 sentences by their specific emotion scores and return  
    def get_top_sentences_by_key(self, input_sentences, emotion, k):
        # Apply the black-box function to each sentence and create a list of dictionaries
        sample_level_scores = dict()
        for sentence in input_sentences:
            escores = emotion_scores(sentence)
            escore = 0.5 
            for d in escores:
                if d['label']==emotion:
                    escore = d['score']
                    break
            sample_level_scores[sentence] = escore    
        sorted_items = sorted(sample_level_scores.items(), key=lambda x: x[1], reverse=True)  
        top_keys = [key for key, value in sorted_items[:k]]
        return top_keys

    # Generate the bigram dictionary     
    def calculate_probability(self):
        for bigram in self.bigramDictionary:
            self.bigramProbabilities[bigram] = (self.bigramDictionary[bigram]) / (self.unigram_counts[bigram[0]])
           
    # Generate next token by standard bigram probability 
    def generate_next_token(self, prev_token):
        if prev_token in self.bigram_counts:
            next_tokens = list(self.bigram_counts[prev_token].keys())
            probabilities = [self.bigramProbabilities[(prev_token, token)] for token in next_tokens]
            next_token = np.random.choice(next_tokens, p=probabilities)
            return next_token

        return None
        
    # Generate given number of sentences by standard bigram porbabilty
    def generate_sentences_standard_bigram(self, num_samples, start_token="$"):
        if len(start_token.split()) ==0:
            start_token = "$"
            
        start_token = start_token.split()[-1]
        for x in range(num_samples):
            prev_token = start_token
            for i in range(10):
                next_token = self.generate_next_token(prev_token)
                print(prev_token, end=" ")
                prev_token = next_token
            print()
        print()
        
with open("corpus.txt", "r", encoding="utf-8") as file:
    corpus = [line.strip().split() for line in file] 

bigram_model = BigramLM()
bigram_model.learn_model(corpus)

