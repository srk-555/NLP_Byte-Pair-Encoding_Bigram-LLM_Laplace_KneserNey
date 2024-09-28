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
            
    # Function to calculate the emotion-oriented bigram probability
    def emotion_bigram_probability(self, bigram, emotion, beta):
        # Get emotion scores for the current word
        # emotion_scores_wi = emotion_scores(bigram[0]+bigram[1])

        # Calculate the emotion component (beta) based on the desired emotion (e.g., joy)
        # emotion_factor = emotion_scores_wi[emotion] if emotion in emotion_scores_wi else 0.0
        
        emotion_factor = 0.5

        for d in self.bigram_emotion_dictionary[bigram][1]:
            if d['label'] == emotion:
                emotion_factor = d['score']
                break

        # Calculate the emotion-oriented bigram probability
        emotion_oriented_bigram_probability = ((1-beta)*self.bigramProbabilities[bigram]) + (beta * emotion_factor)
        return emotion_oriented_bigram_probability
     
    # Function to import previously saved pickle file, in previous defined function
    def import_stored_pickle_file(self):
        # Load the dictionary from the saved file
        with open("bigram_emotion_dictionary.pkl", 'rb') as file:
            self.bigram_emotion_dictionary = pickle.load(file)
      
    # Generate the most probable next token given a specific emotion
    def generate_next_token_by_emotion(self, prev_token, emotion, beta=0.1):
        if prev_token in self.bigram_counts:
            next_tokens = list(self.bigram_counts[prev_token].keys())
            probabilities = np.array([self.emotion_bigram_probability((prev_token, token), emotion, beta) for token in next_tokens])
            
            probabilities = probabilities/probabilities.sum()
            next_token = np.random.choice(next_tokens, p=probabilities)
            return next_token

        return None

    # Function for generating samples corresponding to a specific emotion (e.g., joy)
    def generate_emotion_oriented_samples(self, emotion, beta=0.1, num_samples=5, start_token="$"):
        if len(start_token.split()) ==0:
            start_token = "$"    
        start_token = start_token.split()[-1]
        
        if num_samples>300:
            print("Can generate only upto 300 samples")
            
        if num_samples<=50:
            iterations = num_samples*6
        else:
            iterations = 300
        
        # First store all the generated sentences in list
        generated_sentences = []
        
        # For each iteration, generate 1 sentence
        for _ in range(iterations):
            sample_list = []
            prev_token = start_token
            for i in range(10):
                next_token = self.generate_next_token_by_emotion(prev_token, emotion, beta)
#                 print(prev_token, end=" ")
                sample_list.append(prev_token)
                prev_token = next_token
#             print()
            generated_sentences.append(" ".join(sample_list))
            
        top_k = self.get_top_sentences_by_key(generated_sentences, emotion, num_samples)

        try:
            with open(f"{emotion}.txt", 'w') as file:
                pass  # Doing nothing inside the block, just opening in write mode to clear content
        except Exception as e:
            print(f"Error clearing file: {e}")
        # Save the generated sentences to corresponding text files
        for element in top_k:
            print(element)
            try:
                # Open the file in append mode
                with open(f"{emotion}.txt", 'a') as file:
                    # Append the string to the end of the file
                    file.write(element + '\n')
            except Exception as e:
                print(f"Error writing sample to file: {e}")      
        print()
    
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




with open("corpus.txt", "r", encoding="utf-8") as file:
    corpus = [line.strip().split() for line in file] 

bigram_model = BigramLM()
bigram_model.learn_model(corpus)

