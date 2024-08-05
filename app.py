from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model/recommendation_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the interaction data
data = pd.read_csv('data/interactions.csv')
interaction_matrix = data.pivot_table(index='user_id', columns='video_id', aggfunc='size', fill_value=0)

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    
    if user_id not in interaction_matrix.index:
        return jsonify({'error': 'User not found'}), 404
    
    user_vector = interaction_matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = model.kneighbors(user_vector, n_neighbors=6)
    
    recommendations = []
    for idx in indices[0]:
        if idx != user_id:
            recommendations.append({
                'user_id': interaction_matrix.index[idx],
                'similarity': 1 - distances[0][idx]
            })
    
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)