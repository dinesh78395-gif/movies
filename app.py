from flask import Flask, render_template, request
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)


# Load dataset
movies = pd.read_csv('tmdb_5000_movies.csv')

movies = movies[['title', 'overview', 'genres', 'keywords']]

movies = movies.dropna()

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

vectors = cv.fit_transform(movies['tags']).toarray()


# Similarity matrix
similarity = cosine_similarity(vectors)


# Recommendation function
def recommend(movie):

    movie_index = movies[movies['title'] == movie].index[0]

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

        try:
            recommendations = recommend(movie)

        except:
            recommendations = ["Movie not found"]

    return render_template(
        'index.html',
        recommendations=recommendations
    )


if __name__ == '__main__':
    app.run(debug=True)