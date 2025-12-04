import numpy as np
import os
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource(show_spinner=True)
def load_embeddings(texts):
    if os.path.exists("embeddings.npy"):
        return np.load("embeddings.npy")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(texts, batch_size=16, show_progress_bar=True)
    np.save("embeddings.npy", emb)
    return emb
