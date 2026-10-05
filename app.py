import streamlit as st
from recommender import load_bundle, find_movie, recommend_movies, recommend_for_user

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")
st.caption("KNN on TF-IDF content features · personalised from users' liked movies")

bundle = load_bundle()
movies = bundle["movies"]

tab1, tab2 = st.tabs(["Similar movies", "Personalised for a user"])

with tab1:
    choice = st.selectbox("Search or select a movie", sorted(movies["title"].unique()))
    top_n = st.slider("Number of recommendations", 5, 20, 10)
    if st.button("Recommend"):
        row_number = find_movie(choice, bundle)[0]
        info = movies.loc[row_number]
        st.write(f"**{info['title']}** · {info['genres']} · segment: *{info['cluster_label']}*")
        st.dataframe(recommend_movies(choice, top_n, bundle), hide_index=True)

with tab2:
    user_id = st.selectbox("User ID", sorted(bundle["liked"]))
    n_user = st.slider("Number of recommendations", 5, 20, 10, key="n_user")
    if st.button("Recommend for user"):
        st.caption(f"Based on {len(bundle['liked'][user_id])} movies this user liked (rating ≥ 4)")
        st.dataframe(recommend_for_user(user_id, n_user, bundle), hide_index=True)
