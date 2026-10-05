import streamlit as st
from recommender import load_bundle, find_movie, recommend_movies, recommend_for_user

st.set_page_config(page_title="Personalized Movie Recommender", layout="wide")
st.title("🎬 Personalized Movie Recommendation System")
st.caption("Supervised ML: Random Forest Classifier (P(Like)) + Random Forest Regressor (Rating Prediction)")

bundle = st.cache_resource(load_bundle)()
movies = bundle["movies"]

tab1, tab2 = st.tabs(["Personalised for a User", "Similar Movies"])

with tab1:
    user_id = st.selectbox("Select User ID", sorted(bundle["liked"].keys()))
    n_user = st.slider("Number of recommendations", 5, 20, 10, key="n_user")
    if st.button("Generate Recommendations", type="primary"):
        n_liked = len(bundle["liked"].get(user_id, []))
        st.caption(f"Based on **{n_liked}** movies this user previously liked (rating ≥ 4.0)")
        recs = recommend_for_user(user_id, n_user, bundle)
        st.dataframe(recs, hide_index=True, use_container_width=True)

with tab2:
    choice = st.selectbox("Search or select a movie", sorted(movies["title"].unique()))
    top_n = st.slider("Number of recommendations", 5, 20, 10, key="n_movie")
    if st.button("Find Similar Movies"):
        idx = find_movie(choice, bundle)[0]
        if idx is not None:
            info = movies.loc[idx]
            st.write(f"**{info['title']}** ({info['release_year']}) · {info['genres']} · TMDb Rating: {info['vote_average']}")
            st.dataframe(recommend_movies(choice, top_n, bundle), hide_index=True, use_container_width=True)
