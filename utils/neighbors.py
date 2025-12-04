from sklearn.neighbors import NearestNeighbors
import streamlit as st

@st.cache_resource
def build_neighbor_index(emb, k=11):
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(emb)
    return nn

def get_similar(idx, nn, emb):
    dist, ind = nn.kneighbors([emb[idx]])
    return dist[0][1:], ind[0][1:]
