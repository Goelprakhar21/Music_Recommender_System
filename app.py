import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt')

# -------------------------------
# Load pre-trained models and data
# -------------------------------
@st.cache(allow_output_mutation=True)
def load_data():
    # Load the pickled DataFrame and TF-IDF similarity matrix
    df = pickle.load(open('df.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    return df, similarity

df, similarity = load_data()

# -------------------------------
# TF-IDF Based Song Recommendation
# -------------------------------
def recommend_tfidf(song):
    # Check if song exists in the dataset
    if song not in df['Song'].values:
        st.error(f"Song '{song}' not found in the database.")
        return []
    
    idx = df[df['Song'] == song].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    recommended_songs = []
    # Get top 10 recommendations (skip the first one since it is the input song)
    for i in distances[1:11]:
        recommended_songs.append({
            'Song': df.iloc[i[0]]['Song'],
            'Artist': df.iloc[i[0]]['Artist']
        })
    return recommended_songs

# -------------------------------
# BERT Based Song Recommendation
# -------------------------------
@st.cache(allow_output_mutation=True)
def load_bert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

bert_model = load_bert_model()

def recommend_song_bert(song_name, top_n=10):
    if song_name not in df['Song'].values:
        st.error(f"Song '{song_name}' not found in the database.")
        return pd.DataFrame()
    
    # If the BERT embeddings are not present, compute them and store in df.
    if 'bert_embedding' not in df.columns:
        st.info("Computing BERT embeddings for all songs. This may take a moment...")
        df["bert_embedding"] = df["Text"].apply(lambda x: bert_model.encode(x, convert_to_numpy=True))
    
    song_idx = df[df["Song"] == song_name].index[0]
    song_embedding = df.loc[song_idx, "bert_embedding"].reshape(1, -1)
    all_embeddings = np.vstack(df["bert_embedding"].values)
    similarities = cosine_similarity(all_embeddings, song_embedding).flatten()
    similar_indices = similarities.argsort()[-(top_n+1):-1][::-1]
    return df.iloc[similar_indices][["Song", "Artist"]]

# -------------------------------
# Similar Artists Recommendation
# -------------------------------
@st.cache(allow_output_mutation=True)
def compute_artist_embeddings():
    # Compute or retrieve precomputed BERT embeddings for artists by averaging song embeddings
    if 'bert_embedding' not in df.columns:
        st.info("Computing BERT embeddings for all songs. This may take a moment...")
        df["bert_embedding"] = df["Text"].apply(lambda x: bert_model.encode(x, convert_to_numpy=True))
    artist_embeddings = df.groupby("Artist")["bert_embedding"].apply(
        lambda x: np.mean(np.vstack(x), axis=0)
    ).to_dict()
    return artist_embeddings

artist_embeddings = compute_artist_embeddings()
artist_names = list(artist_embeddings.keys())
artist_matrix = np.vstack(list(artist_embeddings.values()))

def find_similar_artists(artist_name, top_n=5):
    if artist_name not in artist_embeddings:
        st.error(f"Artist '{artist_name}' not found in the database.")
        return []
    artist_idx = artist_names.index(artist_name)
    artist_embedding = artist_embeddings[artist_name].reshape(1, -1)
    similarities = cosine_similarity(artist_matrix, artist_embedding).flatten()
    similar_indices = similarities.argsort()[-(top_n+1):-1][::-1]
    return [artist_names[i] for i in similar_indices]

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.title("Music Recommender System")
st.markdown("""
This interactive app provides two recommendation modes:
- **TF-IDF Based Recommendation:** Finds songs similar to your input song based on text features.
- **BERT Based Recommendation:** Uses deep text embeddings for a more semantic song similarity.
- **Similar Artists:** Discover artists similar to your favorite artist.
""")

# Sidebar for navigation
app_mode = st.sidebar.radio("Choose Recommendation Mode", 
                            ["TF-IDF Song Recommendation", 
                             "BERT Song Recommendation", 
                             "Similar Artists", 
                             "About"])

if app_mode == "TF-IDF Song Recommendation":
    st.header("TF-IDF Based Song Recommendation")
    song_input = st.selectbox("Select or type a song:", df['Song'].unique())
    if st.button("Get Recommendations"):
        recommendations = recommend_tfidf(song_input)
        if recommendations:
            st.subheader("Recommended Songs:")
            for rec in recommendations:
                st.markdown(f"**{rec['Song']}** by *{rec['Artist']}*")
                
elif app_mode == "BERT Song Recommendation":
    st.header("BERT Based Song Recommendation")
    song_input_bert = st.selectbox("Select or type a song:", df['Song'].unique(), key="bert_song")
    if st.button("Find Similar Songs"):
        rec_df = recommend_song_bert(song_input_bert)
        if not rec_df.empty:
            st.subheader("Similar Songs (BERT):")
            st.table(rec_df.reset_index(drop=True))
            
elif app_mode == "Similar Artists":
    st.header("Find Similar Artists")
    artist_input = st.selectbox("Select or type an artist:", sorted(artist_names))
    if st.button("Show Similar Artists"):
        similar = find_similar_artists(artist_input)
        if similar:
            st.subheader(f"Artists similar to **{artist_input}**:")
            for art in similar:
                st.markdown(f"- {art}")
                
elif app_mode == "About":
    st.header("About This App")
    st.markdown("""
    **Music Recommender System**

    This application demonstrates two different recommendation techniques:
    
    1. **TF-IDF Based Recommendation:** Uses TF-IDF vectorization of song texts to compute cosine similarity.
    2. **BERT Based Recommendation:** Leverages deep semantic embeddings from a pre-trained Sentence Transformer model.
    
    The app also features a functionality to find similar artists based on aggregated song embeddings.

    The underlying dataset is processed and pre-trained using a combination of traditional NLP and modern deep learning techniques.
    """)

st.sidebar.markdown("Developed with ❤️ using Streamlit")
