#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Initially define how many emotions we have
emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# For each probable emotion, call the generate_emotion_oriented_samples(emotion, beta, num_of_samples) method from class
# BigramLM() to generate 50 samples for each emotion and save them in .txt files
for emotion_ in emotions:
    print("\nSamples for ", emotion_)
    bigram_model.generate_emotion_oriented_samples(emotion_, 0.7, 50)
    
    
emotion_data = {}
for emotion in emotions:            
    with open(f"{emotion}.txt", "r", encoding="utf-8") as file:
        emotion_data[f"{emotion}_samples"] = [line.strip() for line in file]
        
        
# Combining Data
generated_samples = []
generated_labels = []


for emotion in emotion_data:
    for emotion_sample in emotion_data[emotion]:             
        generated_samples.append(emotion_sample)
        generated_labels.append(emotion.split("_")[0])
        
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Load the original corpus and labels
with open('corpus.txt', 'r', encoding='utf-8') as file:
    corpus = file.read().splitlines()

with open('labels.txt', 'r', encoding='utf-8') as file:
    labels = file.read().splitlines()

# Assuming 'generated_samples' and 'generated_labels' are lists containing the generated samples and labels
# X_train, X_test, y_train, y_test = train_test_split(corpus + generated_samples, labels + generated_labels, test_size=0.2, random_state=42)


X_train = corpus
y_train = labels
X_test = generated_samples
# X_test = ["$ i was feeling doubtful that time i feel like"]
y_test = generated_labels
# y_test = ["fear"]

# Define the pipeline with TF-IDF vectorizer and SVC classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svc', SVC())
])

# # Define parameter grid for Grid Search
# param_grid = {
#     'tfidf__max_features': [5000, 10000, 20000],
#     'svc__C': [0.1, 1, 10],
#     'svc__kernel': ['linear', 'rbf']
# }

# Define parameter grid for Grid Search
param_grid = {
    'tfidf__max_features': [5000, 10000, 20000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],  # Adjust n-gram range
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf'],
    'svc__gamma': ['scale', 'auto', 0.1, 1.0],  # Adjust gamma
    'svc__class_weight': [None, 'balanced']  # Adjust class weight
}

# Perform Grid Search
grid_search = GridSearchCV(pipeline, param_grid, cv=10, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validated Score:", grid_search.best_score_)


# Use the best model to predict on the test set
y_pred_by_model = grid_search.predict(X_test)

for generated_sample in generated_samples:
    maxx_score = 0
    maxx_emotion = ""
    e_scores = emotion_scores(generated_sample)
    for d in e_scores:
        if d['score']>maxx_score:
            maxx_score = d['score']
            maxx_emotion = d['label']
    y_pred_by_emotion_scores.append(maxx_emotion)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred_by_model)
classification_rep = classification_report(y_test, y_pred_by_model)

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n", classification_rep)

