import umap
import streamlit as st

@st.cache_resource(show_spinner=True)
def get_umap_projection(emb):
    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    return reducer.fit_transform(emb)
