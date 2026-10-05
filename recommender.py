import difflib
import joblib
import numpy as np
import pandas as pd

_cache = {}          # remembers the loaded file, so it is not loaded again and again


def load_bundle(path="models/recommender_bundle.joblib"):
    if path not in _cache:
        _cache[path] = joblib.load(path)
    return _cache[path]


def find_movie(query, bundle=None):
    """Return (row number, suggestions). Exact title match first, then partial match; most-rated movie wins."""
    if bundle is None:
        bundle = load_bundle()
    movies = bundle["movies"]
    q = str(query).lower().strip()
    keys = movies["title_key"]

    # 1. exact match
    exact = movies[keys == q]
    if len(exact) > 0:
        return int(exact["n_ratings"].idxmax()), []

    # 2. partial match (query is part of the title)
    if q != "":
        partial = movies[keys.str.contains(q, regex=False)]
        if len(partial) > 0:
            return int(partial["n_ratings"].idxmax()), []

    # 3. not found: suggest close titles
    close = difflib.get_close_matches(q, keys.tolist(), n=5, cutoff=0.5)
    suggestions = []
    for c in close:
        suggestions.append(movies.loc[keys == c, "title"].iloc[0])
    return None, suggestions


def recommend_movies(movie_title, top_n=10, bundle=None):
    """Top-N movies most similar to movie_title (KNN, cosine similarity)."""
    if bundle is None:
        bundle = load_bundle()
    idx, suggestions = find_movie(movie_title, bundle)
    if idx is None:
        message = f"Movie '{movie_title}' not found."
        if len(suggestions) > 0:
            message = message + " Did you mean: " + "; ".join(suggestions)
        print(message)
        return pd.DataFrame()

    dist, ind = bundle["knn_model"].kneighbors(bundle["movie_matrix"][idx], n_neighbors=int(top_n) + 1)
    keep = ind[0] != idx                                   # remove the movie itself from its own list
    dist = dist[0][keep][:top_n]
    ind = ind[0][keep][:top_n]

    recs = bundle["movies"].iloc[ind][["title", "genres", "release_year", "vote_average", "n_ratings"]].copy()
    recs["similarity"] = (1 - dist).round(4)
    recs.insert(0, "Rank", np.arange(1, len(recs) + 1))
    return recs.reset_index(drop=True)


def recommend_for_user(user_id, top_n=10, bundle=None):
    """Top-N movies for a user: neighbours of the movies the user liked (rating >= 4), already-rated movies hidden."""
    if bundle is None:
        bundle = load_bundle()
    seeds = bundle["liked"].get(user_id, [])[-20:]          # last 20 liked movies
    if len(seeds) == 0:
        print(f"User {user_id} not found or has no liked movies.")
        return pd.DataFrame()

    dist, ind = bundle["knn_model"].kneighbors(bundle["movie_matrix"][seeds], n_neighbors=51)
    score = {}
    for d_row, i_row in zip(dist, ind):
        for d, i in zip(d_row, i_row):
            if i not in bundle["seen"][user_id]:
                score[i] = score.get(i, 0) + 1 - d

    top = sorted(score, key=score.get, reverse=True)[:int(top_n)]    # highest score first
    recs = bundle["movies"].iloc[top][["title", "genres", "release_year", "vote_average"]].copy()
    scores_list = []
    for i in top:
        scores_list.append(round(score[i], 3))
    recs["score"] = scores_list
    recs.insert(0, "Rank", np.arange(1, len(recs) + 1))
    return recs.reset_index(drop=True)
