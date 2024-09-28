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
    
    def build_laplace_and_kneserNey_probabilities(self):
        for bigram in self.bigramDictionary:
            self.laplaceProbabilities[bigram] = self.laplace_smoothing_probability(bigram)
            self.kneserneyProbabilities[bigram] = self.kneserney_smoothing_probability(bigram)
            
    
     
    # Function to import previously saved pickle file, in previous defined function
    def import_stored_pickle_file(self):
        # Load the dictionary from the saved file
        with open("bigram_emotion_dictionary.pkl", 'rb') as file:
            self.bigram_emotion_dictionary = pickle.load(file)
      
    
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

    
    # Task 2 : SubTask 2: Implement laplace smoothing
    def laplace_smoothing_probability(self, bigram):
        prefix_count = self.unigram_counts[bigram[0]]
        if bigram not in self.bigramDictionary:
            return ((1)/(prefix_count+len(self.vocab))) 
        
        bigram_count = self.bigramDictionary[bigram]
        return ((bigram_count+1)/(prefix_count+len(self.vocab)))      
        
    # Task 2 : SubTask 2: Implement Kneser Ney smoothing
    def kneserney_smoothing_probability(self, bigram):
        prefix_count = self.unigram_counts[bigram[0]]
        bigram_count = 0
        if bigram in self.bigramDictionary:
            bigram_count = self.bigramDictionary[bigram]
        
        bigram_types_with_suffix = len([x for x in self.bigramDictionary if x[1]==bigram[1]]) # fixed suffix, variable prefix(wi-1)
        bigram_types_with_prefix = len([x for x in self.bigramDictionary if x[0]==bigram[0]]) # fixed prefix, variable suffix(wi)
        total_bigram_types = len(self.bigramDictionary)
        
        discounted_prob = max(bigram_count - self.discount, 0) / prefix_count
        alpha_parameter = (self.discount / prefix_count) * bigram_types_with_prefix
        pcontinuation = bigram_types_with_suffix / total_bigram_types
        
        return discounted_prob + alpha_parameter * pcontinuation
           

    # Generate next token by Laplace probability
    def generate_next_token_using_laplace(self, prev_token):
        next_tokens = [token for token in self.unigram_counts]
        
        probabilities = np.array([self.laplace_smoothing_probability((prev_token, token)) for token in next_tokens])
        
        next_token = np.random.choice(next_tokens, p=probabilities)
        return next_token

    # Generate next token by Kneser Ney probability
    def generate_next_token_using_kneserney(self, prev_token):
        next_tokens = [token for token in self.unigram_counts]
        
        probabilities = np.array([self.kneserney_smoothing_probability((prev_token, token)) for token in next_tokens])
        next_token = np.random.choice(next_tokens, p=probabilities)
        return next_token
        
    # Generate given number of sentences by Laplace porbabilty
    def generate_sentences_laplace(self, num_samples, start_token="$"):
        if len(start_token.split()) ==0:
            start_token = "$"
            
        start_token = start_token.split()[-1]
        for x in range(num_samples):
            prev_token = start_token
            for i in range(10):
                next_token = self.generate_next_token_using_laplace(prev_token)
                print(prev_token, end=" ")
                prev_token = next_token
            print()
        print()
        
    # Generate given number of sentences by KneserNey porbabilty        
    def generate_sentences_kneserney(self, num_samples, start_token="$"):
        if len(start_token.split()) ==0:
            start_token = "$"
            
        start_token = start_token.split()[-1]
        for x in range(num_samples):
            prev_token = start_token
            for i in range(10):
                next_token = self.generate_next_token_using_kneserney(prev_token)
                print(prev_token, end=" ")
                prev_token = next_token
            print()
        print()


with open("corpus.txt", "r", encoding="utf-8") as file:
    corpus = [line.strip().split() for line in file] 

bigram_model = BigramLM()
bigram_model.learn_model(corpus)

