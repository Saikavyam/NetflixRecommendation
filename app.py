import subprocess
import sys
try:
    import neattext.functions as nfx
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "install", "neattext"])
    import neattext.functions as nfx  

import pandas as pd
import numpy as np
import neattext.functions as nfx
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore")


df =  pd.read_csv(r'C:\Users\8saik\Downloads\netflix_titles.csv')
df.head()




df.rename(columns = {'listed_in': 'genres'}, inplace= True)




df['type'].value_counts()





movies_df = df[df['type'] == 'Movie'].reset_index(drop= True)
movies_df.head()





movies_df.duplicated().sum()




movies_df.isnull().sum()





movies_df['rating'].fillna('NaN', inplace= True)
movies_df.dropna(inplace= True)
movies_df = movies_df.reset_index(drop=True)
movies_df.head()





movies = movies_df[['title','director', 'cast', 'country', 'rating', 'genres']]
movies.head()

movies.describe().T

movies['director'] = movies['director'].apply(nfx.remove_stopwords)
movies['cast'] = movies['cast'].apply(nfx.remove_stopwords)
movies['country'] = movies['country'].apply(nfx.remove_stopwords)
movies['genres'] = movies['genres'].apply(nfx.remove_stopwords)

# # Remove special characters
movies['country'] = movies['country'].apply(nfx.remove_special_characters)

movies.head()





countVector = CountVectorizer(binary= True)
country = countVector.fit_transform(movies['country']).toarray()

countVector = CountVectorizer(binary= True,
                             tokenizer=lambda x:x.split(','))
director = countVector.fit_transform(movies['director']).toarray()
cast = countVector.fit_transform(movies['cast']).toarray()
genres = countVector.fit_transform(movies['genres']).toarray()


# In[ ]:


# Turning vectors to dataframe
binary_director = pd.DataFrame(director).transpose()
binary_cast = pd.DataFrame(cast).transpose()
binary_country = pd.DataFrame(country).transpose()
binary_genres = pd.DataFrame(genres).transpose()
# Concating Dataframe
movies_binary = pd.concat([binary_director, binary_cast,  binary_country, binary_genres], axis=0,ignore_index=True)
movies_binary.T

movies_sim = cosine_similarity(movies_binary.T)
movies_sim

movies_sim.shape
# In[ ]:


tv_show = df[df['type'] == 'TV Show'].reset_index(drop= True)
tv_show.head()




tv_show.duplicated().sum()


# In[ ]:


tv_show.isnull().sum()


# In[ ]:


tv_show['director'].fillna('NaN', inplace = True)

# Dropping null values 
tv_show.dropna(inplace= True)
tv_show = tv_show.reset_index(drop=True)
tv_show.head()


# In[ ]:


# Selecting features for working 
tv_df = tv_show[['title','director', 'cast', 'country', 'rating', 'genres']]
tv_df.head()



tv_df.describe().T


tv_df['cast'] = tv_df['cast'].apply(nfx.remove_stopwords)
tv_df['country'] = tv_df['country'].apply(nfx.remove_stopwords)
tv_df['genres'] = tv_df['genres'].apply(nfx.remove_stopwords)

# # Remove special characters
tv_df['country'] = tv_df['country'].apply(nfx.remove_special_characters)

tv_df.head()





countVector = CountVectorizer(binary= True)
country = countVector.fit_transform(tv_df['country']).toarray()

countVector = CountVectorizer(binary= True,
                             tokenizer=lambda x:x.split(','))
cast = countVector.fit_transform(tv_df['cast']).toarray()
genres = countVector.fit_transform(tv_df['genres']).toarray()

tv_binary_cast = pd.DataFrame(cast).transpose()
tv_binary_country = pd.DataFrame(country).transpose()
tv_binary_genres = pd.DataFrame(genres).transpose()
tv_binary = pd.concat([tv_binary_cast,  tv_binary_country, tv_binary_genres], axis=0,ignore_index=True)
tv_binary.T


# In[ ]:


# Concating Dataframe
tv_binary = pd.concat([tv_binary_cast,  tv_binary_country, tv_binary_genres], axis=0,ignore_index=True)
tv_binary.T


# In[ ]:


tv_sim = cosine_similarity(tv_binary.T)
tv_sim


# In[ ]:


tv_sim.shape


# In[ ]:


def recommend(title):
    if title in movies_df['title'].values:
        movies_index = movies_df[movies_df['title'] == title].index.item()
        scores = dict(enumerate(movies_sim[movies_index]))
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        selected_movies_index = [id for id, scores in sorted_scores.items()]
        selected_movies_score = [scores for id, scores in sorted_scores.items()]
        
        rec_movies = movies_df.iloc[selected_movies_index]
        rec_movies['similiarity'] = selected_movies_score

        movie_recommendation = rec_movies.reset_index(drop=True)
        return movie_recommendation[1:6] # Skipping the first row 
    
    elif title in tv_show['title'].values:
        tv_index = tv_show[tv_show['title'] == title].index.item()
        scores = dict(enumerate(tv_sim[tv_index]))
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        selected_tv_index = [id for id, scores in sorted_scores.items()]
        selected_tv_score = [scores for id, scores in sorted_scores.items()]

        rec_tv = tv_show.iloc[selected_tv_index]
        rec_tv['similiarity'] = selected_tv_score

        tv_recommendation = rec_tv.reset_index(drop=True)
        return tv_recommendation[1:6] # Skipping the first row 

    else:
        print("Title not in dataset. Please check spelling.")


# In[ ]:


recommend("Child's Play") #Movie recommendation test 


# In[ ]:


recommend('Bridgerton')


# In[ ]:


import plotly.graph_objects as go


# In[ ]:


def Table(df):
    fig = go.Figure(data=[go.Table(
        columnorder=[1, 2, 3, 4, 5],
        columnwidth=[20, 20, 20, 30, 50],
        header=dict(values=list(['Type', 'Title', 'Country', 'Genre(s)', 'Description']),
                    line_color='black', font=dict(color='black', family="Gravitas One", size=20), height=40,
                    fill_color='#FF6865',
                    align='center'),
        cells=dict(values=[df.type, df.title, df.country, df.genres, df.description],
                   font=dict(color='black', family="Lato", size=16),
                   fill_color='#FFB3B2',
                   align='left'))
    ])

    fig.update_layout(height=700,
                      title={'text': "Top 10 Movie Recommendations", 'font': {'size': 22, 'family': 'Gravitas One'}},
                      title_x=0.5
                      )
    fig.show()


# In[ ]:


Table(recommend('Elite'))


# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import json

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

movies_df = pd.read_csv(r'C:\Users\8saik\Downloads\movies_df.csv')
movies_sim = np.load(r'C:\Users\8saik\Downloads\movies_sim.npz')
movies_sim = movies_sim['m']

tv_show = pd.read_csv(r'C:\Users\8saik\Downloads\tv_show.csv')
tv_sim = np.load(r'C:\Users\8saik\Downloads\tv_sim.npz')
tv_sim = tv_sim['t']

def recommend(title):
    if title in movies_df['title'].values:
        movies_index = movies_df[movies_df['title'] == title].index.item()
        scores = dict(enumerate(movies_sim[movies_index]))
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        selected_movies_index = [id for id, scores in sorted_scores.items()]
        selected_movies_score = [scores for id, scores in sorted_scores.items()]

        rec_movies = movies_df.iloc[selected_movies_index]
        rec_movies['similiarity'] = selected_movies_score

        movie_recommendation = rec_movies.reset_index(drop=True)
        return movie_recommendation[1:11]  # Skipping the first row

    elif title in tv_show['title'].values:
        tv_index = tv_show[tv_show['title'] == title].index.item()
        scores = dict(enumerate(tv_sim[tv_index]))
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        selected_tv_index = [id for id, scores in sorted_scores.items()]
        selected_tv_score = [scores for id, scores in sorted_scores.items()]

        rec_tv = tv_show.iloc[selected_tv_index]
        rec_tv['similiarity'] = selected_tv_score

        tv_recommendation = rec_tv.reset_index(drop=True)
        return tv_recommendation[1:11]  # Skipping the first row



movie_list = sorted(movies_df['title'].tolist() + tv_show['title'].tolist())


st.header('Netflix Movie Recommendation System ')
lottie_coding = load_lottiefile(r"C:\Users\8saik\Downloads\netflix-logo.json")
st_lottie(
    lottie_coding,
    speed=1,
    reverse=False,
    loop=True,
    quality="low",height=220
)
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names = recommend(selected_movie)
    # display table
    st.subheader("Top 10 Recommended Movies")
    st.dataframe(data=recommended_movie_names[['title', 'country', 'genres', 'description', 'release_year', 'cast']])





