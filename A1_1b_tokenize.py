#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
        
    # A fucntion that tokenizes the text based on merge rules
    def tokenize(self, sample: str) -> List[str]:
        actual_tokens = []
        
        # A set to store intermediate tokens
        intermediate_tokens = set()
        
        # Split the snetence into list of words
        sample_list = sample.split()

        for word in sample_list:
            word = word+"$"
            wordpart_list = [] # To keep track of individual characters
            for letter in word:
                wordpart_list.append(letter)
                intermediate_tokens.add(letter)

            # For each rule in merge_rules combine the corresonding pairs of characters in the word
            for rule in self.merge_rules:
                i=0
                while i<len(wordpart_list)-1:
                    if rule[0]+rule[1] == wordpart_list[i]+wordpart_list[i+1]:
                        intermediate_tokens.add(wordpart_list[i]+wordpart_list[i+1])
                        wordpart_list[i] = wordpart_list[i]+wordpart_list[i+1]
                        del wordpart_list[i+1]
                    i = i+1
            # Add the final merged tokens to actual tokens list
            for subpart in wordpart_list:
                actual_tokens.append(subpart)
        print("All found tokens: ", intermediate_tokens)
        
        ### WRITING SAMPLE TOKENS TO TEXT FILE
            
        # Write all tokens found to tokenized_samples.txt for submittables
        try:
            with open("tokenized_samples.txt", "a") as file:
                file.write(",".join(actual_tokens) + "\n")
        except Exception as e:
            print(f"Error writing tokens of sammple to tokenized_samples.txt: {e}") 
        
        
        ### WRITING ALL FOUND TOKENS TO TEXT FILE
            
        # Write all tokens found to tokens.txt for submittables
        tokens = list(intermediate_tokens)
        tokens = sorted(tokens, key=lambda x: (len(x), x))
        for token in tokens:
            try:
                with open("tokens.txt", "a") as file:
                    file.write(token + "\n")
            except Exception as e:
                print(f"Error writing token to tokens.txt: {e}")
             
        print("Tokenized sample: ", actual_tokens)
        return actual_tokens
                


# In[2]:


# Function to construct a sepearte dictionary with bigram_probabilty scores and corresponding emotion scores of each bigram
# and later saving it to pickle for future reference. this function will only be called once and later saved pickle file will be used 
def add_emotion_component_to_bigram_probabilities(self):
    for bigram in self.bigramProbabilities:
        self.bigram_emotion_dictionary[bigram] = [self.bigramProbabilities[bigram], emotion_scores(bigram[0]+" "+bigram[1])]
    # Save the dictionary to a file using pickle
    with open("bigram_emotion_dictionary.pkl", 'wb') as file:
        pickle.dump(self.bigram_emotion_dictionary, file)


# In[ ]:




