import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer


class MovieRecommender:
    GENRE_WEIGHT = 0.6
    CF_WEIGHT = 0.4
    MIN_GENRE_SIM = 0.3

    def __init__(self, movies_path, ratings_path, tags_path, poster_folder, n_movies=1000):
        self.movies_path = movies_path
        self.ratings_path = ratings_path
        self.tags_path = tags_path
        self.poster_folder = poster_folder
        self.n_movies = n_movies

        self.movies_df = None
        self.ratings_df = None
        self.tags_df = None

        self.movie_mapper = {}
        self.movie_inv_mapper = {}
        self.item_similarity = None

        self.genre_mapper = {}
        self.genre_similarity = None

        self._load_data()
        self._build_similarity()

    def _load_data(self):
        self.movies_df = pd.read_csv(self.movies_path)
        self.ratings_df = pd.read_csv(self.ratings_path)
        self.tags_df = pd.read_csv(self.tags_path)

        self.movies_df = self.movies_df.head(self.n_movies).copy()
        selected_movie_ids = self.movies_df["movieId"].tolist()

        self.ratings_df = self.ratings_df[self.ratings_df["movieId"].isin(selected_movie_ids)].copy()
        self.tags_df = self.tags_df[self.tags_df["movieId"].isin(selected_movie_ids)].copy()

        self.movies_df["year"] = self.movies_df["title"].str.extract(r"\((\d{4})\)")
        self.movies_df["clean_title"] = self.movies_df["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True)

        avg_ratings = self.ratings_df.groupby("movieId")["rating"].mean().reset_index()
        avg_ratings.columns = ["movieId", "avg_rating"]

        rating_counts = self.ratings_df.groupby("movieId")["rating"].count().reset_index()
        rating_counts.columns = ["movieId", "rating_count"]

        self.movies_df = self.movies_df.merge(avg_ratings, on="movieId", how="left")
        self.movies_df = self.movies_df.merge(rating_counts, on="movieId", how="left")

        self.movies_df["avg_rating"] = self.movies_df["avg_rating"].fillna(0).round(1)
        self.movies_df["rating_count"] = self.movies_df["rating_count"].fillna(0).astype(int)
        self.movies_df["genres"] = self.movies_df["genres"].fillna("(no genres listed)")
        self.movies_df["genre_list"] = self.movies_df["genres"].apply(
            lambda g: [] if g == "(no genres listed)" else g.split("|")
        )

    def _build_similarity(self):
        self._build_genre_similarity()

        if self.ratings_df.empty:
            self.item_similarity = np.array([[]])
            return

        # Adjusted cosine similarity: rating tiap user dikurangi rata-rata
        # rating user tsb dulu, supaya bias "user royal kasih nilai tinggi"
        # vs "user pelit kasih nilai" tidak mendistorsi kemiripan antar film
        # (Sarwar et al., 2001).
        user_means = self.ratings_df.groupby("userId")["rating"].mean()
        adjusted_ratings = self.ratings_df.copy()
        adjusted_ratings["rating"] = adjusted_ratings["rating"] - adjusted_ratings["userId"].map(user_means)

        user_movie_matrix = adjusted_ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
            fill_value=0
        )

        movie_ids = list(user_movie_matrix.columns)
        self.movie_mapper = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
        self.movie_inv_mapper = {idx: movie_id for idx, movie_id in enumerate(movie_ids)}

        sparse_matrix = csr_matrix(user_movie_matrix.values)
        item_user_matrix = sparse_matrix.T
        self.item_similarity = cosine_similarity(item_user_matrix)

    def _build_genre_similarity(self):
        """Cosine similarity antar film berdasarkan genre (one-hot). Selalu
        bisa dihitung untuk seluruh film di katalog, walau film itu belum
        pernah dirating siapa pun -- ini yang jadi fallback saat data CF
        kosong/terlalu tipis."""
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(self.movies_df["genre_list"])
        self.genre_similarity = cosine_similarity(genre_matrix)
        self.genre_mapper = {
            movie_id: pos for pos, movie_id in enumerate(self.movies_df["movieId"])
        }

    def get_poster_url(self, movie_id):
        extensions = [".jpg", ".jpeg", ".png", ".webp"]

        for ext in extensions:
            file_path = os.path.join(self.poster_folder, f"{movie_id}{ext}")
            if os.path.exists(file_path):
                return f"/static/posters/{movie_id}{ext}"

        defaults = ["default.jpg", "default.jpeg", "default.png", "default.webp"]
        for default_file in defaults:
            default_path = os.path.join(self.poster_folder, default_file)
            if os.path.exists(default_path):
                return f"/static/posters/{default_file}"

        return "[via.placeholder.com](https://via.placeholder.com/300x450?text=No+Poster)"

    def get_movie_by_id(self, movie_id):
        result = self.movies_df[self.movies_df["movieId"] == movie_id]
        if result.empty:
            return None
        return result.iloc[0].to_dict()

    def get_all_genres(self):
        genres_set = set()

        for genres in self.movies_df["genres"].dropna():
            for genre in str(genres).split("|"):
                genre = genre.strip()
                if genre and genre != "(no genres listed)":
                    genres_set.add(genre)

        return sorted(list(genres_set))

    def search_movies(self, query=None, genre="all", limit=50):
        df = self.movies_df.copy()

        if query:
            df = df[df["title"].str.contains(query, case=False, na=False)]

        if genre and genre.lower() != "all":
            df = df[df["genres"].str.contains(genre, case=False, na=False)]

        df = df.sort_values(
            by=["avg_rating", "rating_count", "title"],
            ascending=[False, False, True]
        ).head(limit)

        return df.to_dict(orient="records")

    def get_similar_movies(self, movie_id, n=10):
        if movie_id not in self.genre_mapper:
            return []

        target_pos = self.genre_mapper[movie_id]
        target_cf_idx = self.movie_mapper.get(movie_id)

        candidates = []
        for other_id, other_pos in self.genre_mapper.items():
            if other_id == movie_id:
                continue

            genre_sim = self.genre_similarity[target_pos][other_pos]
            if genre_sim < self.MIN_GENRE_SIM:
                continue

            cf_sim = 0.0
            other_cf_idx = self.movie_mapper.get(other_id)
            if target_cf_idx is not None and other_cf_idx is not None:
                cf_sim = self.item_similarity[target_cf_idx][other_cf_idx]

            final_score = self.GENRE_WEIGHT * genre_sim + self.CF_WEIGHT * cf_sim
            candidates.append((other_id, final_score, genre_sim, cf_sim))

        candidates.sort(key=lambda x: x[1], reverse=True)

        similar_movies = []
        for other_id, final_score, genre_sim, cf_sim in candidates[:n]:
            similar_movies.append({
                "movieId": int(other_id),
                "similarity": float(round(final_score, 4)),
                "genre_similarity": float(round(genre_sim, 4)),
                "cf_similarity": float(round(cf_sim, 4))
            })

        return similar_movies

    def get_recommendations_for_user(self, liked_movie_ids, n=12):
        if not liked_movie_ids:
            return []

        score_dict = {}

        for movie_id in liked_movie_ids:
            similar_movies = self.get_similar_movies(movie_id, n=20)

            for item in similar_movies:
                sim_movie_id = item["movieId"]
                sim_score = item["similarity"]

                if sim_movie_id in liked_movie_ids:
                    continue

                if sim_movie_id not in score_dict:
                    score_dict[sim_movie_id] = 0

                score_dict[sim_movie_id] += sim_score

        sorted_recommendations = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:n]

        results = []
        for movie_id, total_score in sorted_recommendations:
            movie = self.get_movie_by_id(movie_id)
            if movie:
                movie["recommendation_score"] = round(total_score, 4)
                results.append(movie)

        return results
