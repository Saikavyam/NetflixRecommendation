import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import re

# ✅ Load JSON animation
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# ✅ Text Cleaning Functions (Replacing neattext)
def remove_stopwords(text):
    stopwords = set(["the", "and", "in", "of", "to", "a", "is", "for", "on"])  # Expand this list as needed
    return " ".join([word for word in str(text).split() if word.lower() not in stopwords])

def remove_special_characters(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", str(text))

# ✅ Load Data (Relative Paths)
movies_df = pd.read_csv('movies_df.csv')
tv_show = pd.read_csv('tv_show.csv')
movies_sim = np.load('movies_sim.npz')['m']
tv_sim = np.load('tv_sim.npz')['t']

# ✅ Apply Text Cleaning
movies_df['director'] = movies_df['director'].fillna("").apply(remove_stopwords).apply(remove_special_characters)
movies_df['cast'] = movies_df['cast'].fillna("").apply(remove_stopwords).apply(remove_special_characters)
movies_df['country'] = movies_df['country'].fillna("").apply(remove_stopwords).apply(remove_special_characters)
movies_df['genres'] = movies_df['genres'].fillna("").apply(remove_stopwords).apply(remove_special_characters)

tv_show['director'] = tv_show['director'].fillna("").apply(remove_stopwords).apply(remove_special_characters)
tv_show['cast'] = tv_show['cast'].fillna("").apply(remove_stopwords).apply(remove_special_characters)
tv_show['country'] = tv_show['country'].fillna("").apply(remove_stopwords).apply(remove_special_characters)
tv_show['genres'] = tv_show['genres'].fillna("").apply(remove_stopwords).apply(remove_special_characters)

# ✅ Recommendation Function
def recommend(title):
    if title in movies_df['title'].values:
        movies_index = movies_df[movies_df['title'] == title].index.item()
        scores = dict(enumerate(movies_sim[movies_index]))
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        selected_movies_index = list(sorted_scores.keys())[1:11]  # Skipping the first one
        rec_movies = movies_df.iloc[selected_movies_index].copy()
        rec_movies['similarity'] = list(sorted_scores.values())[1:11]
        return rec_movies[['title', 'country', 'genres', 'description', 'release_year', 'cast', 'similarity']]

    elif title in tv_show['title'].values:
        tv_index = tv_show[tv_show['title'] == title].index.item()
        scores = dict(enumerate(tv_sim[tv_index]))
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        selected_tv_index = list(sorted_scores.keys())[1:11]
        rec_tv = tv_show.iloc[selected_tv_index].copy()
        rec_tv['similarity'] = list(sorted_scores.values())[1:11]
        return rec_tv[['title', 'country', 'genres', 'description', 'release_year', 'cast', 'similarity']]

    else:
        return pd.DataFrame({'title': ['Not Found'], 'country': ['-'], 'genres': ['-'], 
                             'description': ['Title not in dataset. Please check spelling.'],
                             'release_year': ['-'], 'cast': ['-'], 'similarity': ['-']})

# ✅ Streamlit UI
st.header('🎬 Netflix Movie Recommendation System')

# ✅ Load & Display Animation
lottie_coding = load_lottiefile("netflix-logo.json")
st_lottie(lottie_coding, speed=1, loop=True, quality="low", height=220)

# ✅ Dropdown for Movie Selection
movie_list = sorted(movies_df['title'].dropna().tolist() + tv_show['title'].dropna().tolist())
selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

# ✅ Show Recommendations on Button Click
if st.button('Show Recommendation'):
    recommended_movies = recommend(selected_movie)
    st.subheader("🔝 Top 10 Recommended Movies/Shows")
    st.dataframe(recommended_movies)
