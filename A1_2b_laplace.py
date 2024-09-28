#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


# Task2: Subtask2 : Function to compare and return the Laplace and kneserNey probabilities
def compare_probabilities(self, prev_token="$"):
    if len(prev_token.split()) ==0:
        prev_token = "$"

    prev_token = prev_token.split()[-1]
    if prev_token in self.bigram_counts:
        available_next_tokens = list(self.bigram_counts[prev_token].keys())
        probabilities_bigram = [self.bigramProbabilities[(prev_token, token)] for token in available_next_tokens]

        all_tokens = np.array([token for token in self.unigram_counts])
        probabilities_laplace = np.array([self.laplace_smoothing_probability((prev_token, token)) for token in self.unigram_counts])
        probabilities_kneserney = np.array([self.kneserney_smoothing_probability((prev_token, token)) for token in self.unigram_counts])

        # Set display options for float formatting
        pd.set_option('display.float_format', '{:.8f}'.format)
        # Create Probability DataFrame
        self.probability_comparison_frame = pd.DataFrame({
            'Token': all_tokens,
            'Laplace Probability': probabilities_laplace,
            'KneserNey Probability': probabilities_kneserney
        })

        self.probability_comparison_frame_available = self.probability_comparison_frame[self.probability_comparison_frame["Token"].isin(available_next_tokens)]
        self.probability_comparison_frame_available["Bigram Probability"] = probabilities_bigram
        print("Probability Comparison of All Tokens")
        print(self.probability_comparison_frame)

        print("\nProbability Comparison of Next Available Tokens in Bigram Dictionary")
        print(self.probability_comparison_frame_available)
        return

    return None


# In[ ]:




