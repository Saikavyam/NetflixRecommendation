from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load Data
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

movies_df = pd.read_csv(os.path.join(BASE_DIR, 'movies_df.csv'))
tv_show = pd.read_csv(os.path.join(BASE_DIR, 'tv_show.csv'))
movies_sim = np.load(os.path.join(BASE_DIR, 'movies_sim.npz'))['m']
tv_sim = np.load(os.path.join(BASE_DIR, 'tv_sim.npz'))['t']
# Recommendation Function
def recommend(title):
    if title in movies_df['title'].values:
        movies_index = movies_df[movies_df['title'] == title].index.item()
        scores = dict(enumerate(movies_sim[movies_index]))
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        selected_movies_index = list(sorted_scores.keys())[1:6]  # Skipping the first one
        rec_movies = movies_df.iloc[selected_movies_index].copy()
        rec_movies['similarity'] = list(sorted_scores.values())[1:6]
        return rec_movies[['title', 'country', 'genres', 'description', 'release_year', 'cast']].to_dict(orient='records')

    elif title in tv_show['title'].values:
        tv_index = tv_show[tv_show['title'] == title].index.item()
        scores = dict(enumerate(tv_sim[tv_index]))
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        selected_tv_index = list(sorted_scores.keys())[1:6]
        rec_tv = tv_show.iloc[selected_tv_index].copy()
        rec_tv['similarity'] = list(sorted_scores.values())[1:6]
        return rec_tv[['title', 'country', 'genres', 'description', 'release_year', 'cast']].to_dict(orient='records')

    else:
        return [{"error": "Title not found"}]

# API Endpoint
@app.route('/recommend', methods=['GET'])
def get_recommendation():
    title = request.args.get('title')
    if not title:
        return jsonify({"error": "Please provide a title parameter"}), 400
    return jsonify(recommend(title))

# Run App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

