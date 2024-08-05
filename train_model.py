import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle

# Load data
data = pd.read_csv('data/interactions.csv')

# Pivot data to create user-item interaction matrix
interaction_matrix = data.pivot_table(index='user_id', columns='video_id', aggfunc='size', fill_value=0)

# Train a simple KNN model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(interaction_matrix)

# Save the model
with open('model/recommendation_model.pkl', 'wb') as f:
    pickle.dump(model, f)