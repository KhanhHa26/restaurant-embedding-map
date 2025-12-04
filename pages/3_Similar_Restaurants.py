import streamlit as st
from utils.data_loader import load_yelp_restaurants, combine_text
from utils.embeddings import load_embeddings
from utils.neighbors import build_neighbor_index, get_similar
from utils.ui_components import restaurant_card
from utils.slider import show_slider_and_confirm_button


# -----------------------------------------------------------
# PAGE HEADER
# -----------------------------------------------------------
st.markdown("""
<h1 class="section-title">✨ Similar Restaurants</h1>
<p class="subtext">
Find restaurants that closely resemble your selected choice based on 
text embeddings derived from descriptions, categories, and attributes.
<br><br>
Similarity is calculated using <strong>cosine distance</strong> between embedding vectors.
</p>
<hr>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# INITIALIZE SESSION STATE
# -----------------------------------------------------------
if "ranked_restaurants" not in st.session_state:
    st.session_state["ranked_restaurants"] = []


# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
restaurants = load_yelp_restaurants()
restaurants["text"] = restaurants.apply(combine_text, axis=1)
embeddings = load_embeddings(restaurants["text"])
nn = build_neighbor_index(embeddings)


# -----------------------------------------------------------
# SELECT RESTAURANT
# -----------------------------------------------------------
st.markdown("""
<h3 class="section-subtitle">📍 Choose a Restaurant</h3>
<p class="subtext">Select a restaurant to discover others that are most similar in style, cuisine, and textual attributes.</p>
""", unsafe_allow_html=True)

options = {
    f"{row['name']} ({row['city']}, {row['state']})": i
    for i, row in restaurants.iterrows()
}

user_choice = st.selectbox(
    "Select a restaurant to compare:", 
    list(options.keys()),
    help="Start typing a name to narrow down your choices."
)
idx = options[user_choice]


# -----------------------------------------------------------
# SHOW SELECTED RESTAURANT (CARD)
# -----------------------------------------------------------
st.markdown("""
<h3 class="section-subtitle">🍽️ Selected Restaurant</h3>
<p class="subtext">This is the restaurant you chose. Scroll down to see similar alternatives.</p>
""", unsafe_allow_html=True)

restaurant_card(restaurants, idx)


# -----------------------------------------------------------
# GET SIMILAR RESTAURANTS
# -----------------------------------------------------------
distances, indices = get_similar(idx, nn, embeddings)


# -----------------------------------------------------------
# DISPLAY SIMILAR RESTAURANTS
# -----------------------------------------------------------
st.markdown("""
<br>
<h3 class="section-subtitle">✨ Recommended Similar Restaurants</h3>
<p class="subtext">
Here are the restaurants most closely related to your selection.
Use the rating slider to add any of them to your personal ranking list.
</p>
""", unsafe_allow_html=True)
left, right = st.columns([1.5, 1.5])
for sim_idx in indices:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    with left:
        restaurant_card(restaurants, sim_idx)
    with right:
        show_slider_and_confirm_button(sim_idx)
    st.markdown("</div>", unsafe_allow_html=True)
