'''
source venv_name/bin/activate 
venv/bin/python -m streamlit run app.py
'''
import numpy as np
import os
import pandas as pd
import plotly.express as px
import streamlit as st
import umap
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

import kagglehub
from PIL import Image
from kagglehub import KaggleDatasetAdapter

# ============================================
# Streamlit Page Config
# ============================================
im = Image.open("restaurant_icon.png")
st.set_page_config(
    page_title="Restaurant Embedding Explorer",
    page_icon=im,
    layout="wide"
)

st.markdown("""
<h1 style='text-align:center; margin-bottom: -10px;'>🍽️ Restaurant Embedding Explorer</h1>
<p style='text-align:center; font-size:18px; color:#666'>
Explore Yelp restaurants visually and discover similar restaurants based on embeddings.
</p>
""", unsafe_allow_html=True)

st.write("")


# ============================================
# 🌐 Sidebar
# ============================================
with st.sidebar:
    st.header("⚙️ Description")
    st.write("This app uses **sentence embeddings + UMAP + nearest neighbors** to visualize and recommend restaurants.")

    with st.expander("How It Works"):
        st.markdown("""
        - Yelp business data → text description  
        - Convert to embeddings using **SentenceTransformers**  
        - Dimensionality reduction using **UMAP**  
        - Interactive visual map  
        - Nearest neighbors to find similar restaurants  
        """)

    st.info("Tip: Try typing part of a restaurant’s name")


# ============================================
# Load Yelp Data
# ============================================
@st.cache_data(show_spinner=True)
def load_yelp_restaurants():
    file_path = "yelp_academic_dataset_business.json"
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "yelp-dataset/yelp-dataset",
        file_path,
        pandas_kwargs={"lines": True}
    )
    r = df[df["categories"].str.contains("Restaurants", na=False)].copy()
    r.reset_index(drop=True, inplace=True)
    return r

with st.spinner("📥 Loading Yelp restaurants..."):
    restaurants = load_yelp_restaurants()
st.success(f"Loaded **{len(restaurants):,}** restaurants 🍜")


# ============================================
# Combine Text Fields
# ============================================
def combine_text(row):
    fields = []
    for col in ["name", "categories", "attributes", "city"]:
        if pd.notna(row[col]):
            fields.append(str(row[col]))
    return " ".join(fields)

restaurants["text"] = restaurants.apply(combine_text, axis=1)


# ============================================
# 🔡 Embeddings
# ============================================
@st.cache_resource(show_spinner=True)
def get_embeddings(texts):
    if os.path.exists("embeddings.npy"):
        st.info("Using cached embeddings.")
        return np.load("embeddings.npy")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(texts, batch_size=16, show_progress_bar=True)
    np.save("embeddings.npy", emb)
    return emb

with st.spinner("⚙️ Generating embeddings (first time only)..."):
    embeddings = get_embeddings(restaurants["text"])

st.success("Embeddings ready!")


# ============================================
# 🔍 Nearest Neighbors
# ============================================
@st.cache_resource
def get_neighbors(emb, k=11):
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(emb)
    return nn

nn = get_neighbors(embeddings)


# ============================================
# 📉 UMAP
# ============================================
@st.cache_resource(show_spinner=True)
def get_umap(emb):
    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    return reducer.fit_transform(emb)

with st.spinner("🎨 Running UMAP… This may take a moment"):
    X_umap = get_umap(embeddings)


# ============================================
# 🗺️ Plot Map
# ============================================
restaurants["main_category"] = restaurants["categories"].apply(
    lambda x: x.split(",")[0].strip() if isinstance(x, str) else "Other"
)

fig = px.scatter(
    restaurants,
    x=X_umap[:, 0],
    y=X_umap[:, 1],
    color="main_category",
    hover_data=["name", "stars", "city"],
    opacity=0.7,
)

st.subheader("📍 Restaurant Embedding Map (Interactive)")
st.caption("Closer points represent restaurants that are more similar.")

st.plotly_chart(fig, use_container_width=True)

# ============================================
# 🔍 Find a Single Restaurant
# ============================================
st.markdown("---")
st.subheader("🔎 Find A Restaurant and Rank")
options = {
    f"{row['name']} ({row['city']}, {row['state']})": i
    for i, row in restaurants.iterrows()
}

user_input_single_restaurant = st.selectbox("Search restaurant:", list(options.keys()))
idx = options[user_input_single_restaurant]

with st.container():    
    
    st.markdown(f"""
    <div style="padding:15px; border-radius:10px; background:#fafafa; margin-bottom:10px; border:1px solid #eee; color: #111111">
        <h4 style="margin:0;">{restaurants.iloc[idx]['name']}</h4>
        <p style="margin:3px 0;">⭐ {restaurants.iloc[idx]['stars']} stars — {restaurants.iloc[idx]['city']}, {restaurants.iloc[idx]['state']}</p>
        <p style="margin:3px 0;">
            <strong>Categories:</strong> {restaurants.iloc[idx]['categories']}
        </p>
    </div>
    """, unsafe_allow_html=True)

#If click on rank button, show the slider 
def handle_button(idx, key):
    state_key = f"rank_{key}_{idx}_open"

    # Button toggles only this specific restaurant's state
    if st.button("Rank this restaurant", key=f"rank_btn_{key}_{idx}"):
        st.session_state[state_key] = not st.session_state.get(state_key, False)

    # Show slider ONLY for this item
    if st.session_state.get(state_key, False):
        score = st.slider(
            "Rank it from 1 to 10",
            0.0, 10.0, 5.0, 0.1,
            key=f"slider_{key}_{idx}"
        )
        st.write(f"You selected: {score}")

handle_button(idx, "single")


# ============================================
# User Search Similar Restaurants
# ============================================
st.markdown("---")
st.subheader("🔎 Find Similar Restaurants")
st.caption("Discover 10 restaurants that are most similar to the one you choose.")

user_input = st.selectbox("Search restaurant:", options, key="similar")
idx = options[user_input]

# def find_restaurant(name, df):
#     name = name.lower().split("(")[0]
#     matches = df[df["name"].str.lower().str.contains(name)]

#     if len(matches) == 0:
#         return None

#     return matches.index[0]

def get_similar_restaurants(user_text, embeddings, df, nn, idx):
    if not user_text:
        return None, None

    # idx = find_restaurant(user_text, df)
    if idx is None:
        return None, None

    distances, indices = nn.kneighbors([embeddings[idx]])
    similar_idx = indices[0][1:]
    similar_distances = distances[0][1:]

    results = df.iloc[similar_idx].copy()
    results["similarity"] = (1 - similar_distances) * 100

    return df.loc[idx, "name"], results


name, results = get_similar_restaurants(user_input, embeddings, restaurants, nn, idx)

# ============================================
# Display Results
# ============================================


def pretty_print2(recommendations):
    rec = recommendations.reset_index(drop=True)

    # Build entire HTML grid as ONE string
    html_cards = """<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 25px; margin-bottom: 20px;">"""

    for i, row in rec.iterrows():
        html_cards += f"""<div style="background: #fafafa; border: 1px solid #eee; border-radius: 12px; padding: 15px; color: #111;">
            <h4>{i+1}. {row['name']}</h4>
            <p>⭐ {row['stars']} stars — {row['city']}, {row['state']}</p>
            <p><strong>Categories:</strong> {row['categories']}</p>
            <p><strong>Similarity:</strong> {row['similarity']:.2f}%</p>
        </div>
        """
    html_cards += "</div>"

    # Render ONCE
    st.markdown(html_cards, unsafe_allow_html=True)
    handle_button(i, "similar")

def pretty_print(recommendations):
    rec = recommendations.reset_index(drop=True)

    for i, row in rec.iterrows():
        st.markdown(f"""
        <div style="padding:12px; border-radius:10px; background:#fafafa; margin-bottom:10px; border:1px solid #eee; color: #111111;">
            <h4 style="margin:0;">{i+1}. {row['name']}</h4>
            <p style="margin:3px 0;">⭐ {row['stars']} stars — {row['city']}, {row['state']}</p>
            <p style="margin:3px 0;">
                <strong>Categories:</strong> {row['categories']}
            </p>
            <p style="margin:3px 0;">
                <strong>Similarity:</strong> {row['similarity']:.2f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
        handle_button(i, "similar")

if user_input.strip():
    if name is None:
        st.error("❌ No matching restaurant found.")
    else:
        st.success(f"🎉 Found restaurants similar to **{name}**:")
        pretty_print(results)

