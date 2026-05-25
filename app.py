from flask import Flask, render_template, request
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)


# Load dataset
movies = pd.read_csv('tmdb_5000_movies.csv')

# Select important columns
movies = movies[['title', 'overview', 'genres', 'keywords']]

# Remove missing values
movies = movies.dropna()

# Create tags column
movies['tags'] = (
    movies['overview'] +
    movies['genres'] +
    movies['keywords']
)

# Convert text into vectors
cv = CountVectorizer(
    max_features=5000,
    stop_words='english'
)

vectors = cv.fit_transform(movies['tags'])

# Similarity matrix
similarity = cosine_similarity(vectors)


# Recommendation function
def recommend(movie):

    movie = movie.lower()

    movies['title_lower'] = movies['title'].str.lower()

    if movie not in movies['title_lower'].values:
        return ["Please enter the full name of the movie correctly"]

    movie_index = movies[movies['title_lower'] == movie].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_list:
        recommended_movies.append(
            movies.iloc[i[0]].title
        )

    return recommended_movies


# Home route
@app.route('/', methods=['GET', 'POST'])
def home():

    recommendations = []

    if request.method == 'POST':

        movie = request.form['movie']

        recommendations = recommend(movie)

    return render_template(
        'index.html',
        recommendations=recommendations
    )


if __name__ == '__main__':
    
    import os

    port = int(os.environ.get("PORT", 5000))

    app.run(host='0.0.0.0', port=port)