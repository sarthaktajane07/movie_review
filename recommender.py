import difflib
import joblib
import numpy as np
import pandas as pd

_cache = {}

def load_bundle(path="models/recommender_bundle.joblib"):
    if path not in _cache:
        _cache[path] = joblib.load(path)
    return _cache[path]

def find_movie(query, bundle=None):
    """Return (row index, suggestions). Exact match first, then partial; most-rated movie wins."""
    b = bundle or load_bundle()
    movies = b["movies"]
    q, keys = str(query).lower().strip(), movies["title_key"]
    for hit in (movies[keys == q], movies[keys.str.contains(q, regex=False)] if q else movies.iloc[:0]):
        if len(hit):
            return int(hit["n_ratings"].idxmax()), []
    close = difflib.get_close_matches(q, keys.tolist(), n=5, cutoff=0.5)
    return None, [movies.loc[keys == c, "title"].iloc[0] for c in close]

def recommend_movies(movie_title, top_n=10, bundle=None):
    """Top-N similar movies based on content metadata and popularity."""
    b = bundle or load_bundle()
    idx, suggestions = find_movie(movie_title, b)
    movies = b["movies"]
    if idx is None:
        print(f"Movie '{movie_title}' not found." + (" Did you mean: " + "; ".join(suggestions) if suggestions else ""))
        return pd.DataFrame()
    
    target_movie = movies.iloc[idx]
    # Simple content/genre similarity ranking
    same_genre = movies[movies.genres == target_movie.genres].copy()
    same_genre = same_genre[same_genre.movieId != target_movie.movieId]
    recs = same_genre.sort_values("n_ratings", ascending=False).head(top_n)
    
    if len(recs) < top_n:
        more = movies[~movies.movieId.isin(set(recs.movieId) | {target_movie.movieId})].sort_values("n_ratings", ascending=False).head(top_n - len(recs))
        recs = pd.concat([recs, more])
        
    recs = recs[["title", "genres", "release_year", "vote_average", "n_ratings"]].copy()
    recs.insert(0, "Rank", np.arange(1, len(recs) + 1))
    return recs.reset_index(drop=True)

def recommend_for_user(user_id, top_n=10, bundle=None):
    """Personalised Top-N recommendation for a user using Best Classifier & Linear Regressor models."""
    b = bundle or load_bundle()
    movies = b["movies"]
    all_movie_ids = movies["movieId"].values
    seen = b["seen"].get(user_id, set())
    
    cand_ids = [m for m in all_movie_ids if m not in seen]
    if not cand_ids:
        return pd.DataFrame()
    
    u_info = b["u_stats"].get(user_id, {"u_cnt": 0, "u_mean": b["global_rating"], "u_like": b["global_like"]})
    movie_meta_dict = b["movie_meta_dict"]
    
    user_genre_profiles = b.get("user_genre_profiles", {})
    movie_g_map = b.get("movie_g_map", {})
    zero_g = b.get("zero_g", np.zeros(1))
    u_prof = user_genre_profiles.get(user_id, zero_g)
    
    rows = []
    for m in cand_ids:
        m_info = b["m_stats"].get(m, {"m_cnt": 0, "m_mean": b["global_rating"], "m_like": b["global_like"]})
        meta_info = movie_meta_dict[m]
        m_prof = movie_g_map.get(m, zero_g)
        g_match = float(np.dot(u_prof, m_prof)) if len(u_prof) == len(m_prof) else 0.0
        
        row = {
            "u_cnt": u_info["u_cnt"], "u_mean": u_info["u_mean"], "u_like": u_info["u_like"],
            "m_cnt": m_info["m_cnt"], "m_mean": m_info["m_mean"], "m_like": m_info["m_like"],
            "vote_average": meta_info.get("vote_average", 6.0),
            "vote_count": meta_info.get("vote_count", 100),
            "release_year": meta_info.get("release_year", 2000),
            "runtime": meta_info.get("runtime", 100),
            "genre_match": g_match
        }
        rows.append(row)
        
    cand_df = pd.DataFrame(rows, columns=b["feature_cols"]).fillna(0)
    prob_like = b["rf_cls"].predict_proba(cand_df)[:, 1]
    pred_rating = np.clip(b["rf_reg"].predict(cand_df), 0.5, 5.0)
    
    rec_score = 0.5 * prob_like + 0.5 * (pred_rating / 5.0)
    
    res = pd.DataFrame({
        "movieId": cand_ids,
        "prob_like": prob_like.round(4),
        "pred_rating": pred_rating.round(2),
        "rec_score": rec_score.round(4)
    }).sort_values("rec_score", ascending=False).head(top_n)
    
    res = res.merge(movies[["movieId", "title", "genres", "release_year"]], on="movieId")
    res.insert(0, "Rank", np.arange(1, len(res) + 1))
    return res[["Rank", "title", "genres", "release_year", "prob_like", "pred_rating", "rec_score"]]

