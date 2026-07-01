from flask import Flask, render_template, request, jsonify
from recommender import MovieRecommender
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
POSTER_DIR = os.path.join(BASE_DIR, "static", "posters")

recommender = MovieRecommender(
    movies_path=os.path.join(DATA_DIR, "movies.csv"),
    ratings_path=os.path.join(DATA_DIR, "ratings.csv"),
    tags_path=os.path.join(DATA_DIR, "tags.csv"),
    poster_folder=POSTER_DIR,
    n_movies=1000
)

@app.route("/")
def index():
    genres = recommender.get_all_genres()
    movies = recommender.search_movies(limit=18)

    for movie in movies:
        movie["poster_url"] = recommender.get_poster_url(movie["movieId"])

    return render_template("index.html", genres=genres, movies=movies)


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    genre = request.args.get("genre", "all").strip()

    movies = recommender.search_movies(
        query=query if query else None,
        genre=genre if genre else "all",
        limit=60
    )

    for movie in movies:
        movie["poster_url"] = recommender.get_poster_url(movie["movieId"])

    return jsonify({
        "success": True,
        "count": len(movies),
        "movies": movies
    })


@app.route("/movie/<int:movie_id>")
def movie_detail(movie_id):
    movie = recommender.get_movie_by_id(movie_id)

    if not movie:
        return jsonify({
            "success": False,
            "message": "Film tidak ditemukan"
        }), 404

    movie["poster_url"] = recommender.get_poster_url(movie["movieId"])

    similar_ids = recommender.get_similar_movies(movie_id, n=8)
    similar_movies = []

    for item in similar_ids:
        sim_movie = recommender.get_movie_by_id(item["movieId"])
        if sim_movie:
            sim_movie["poster_url"] = recommender.get_poster_url(sim_movie["movieId"])
            sim_movie["similarity"] = item["similarity"]
            similar_movies.append(sim_movie)

    return jsonify({
        "success": True,
        "movie": movie,
        "similar_movies": similar_movies
    })


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    liked_movie_ids = data.get("liked_movie_ids", [])

    if not liked_movie_ids:
        return jsonify({
            "success": False,
            "message": "Pilih minimal 1 film"
        }), 400

    recommendations = recommender.get_recommendations_for_user(liked_movie_ids, n=12)

    for movie in recommendations:
        movie["poster_url"] = recommender.get_poster_url(movie["movieId"])

    return jsonify({
        "success": True,
        "recommendations": recommendations
    })


if __name__ == "__main__":
    app.run(debug=True)
