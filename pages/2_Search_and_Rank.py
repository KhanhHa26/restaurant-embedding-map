import streamlit as st
from utils.data_loader import load_yelp_restaurants
from utils.ui_components import restaurant_card
from utils.slider import show_slider_and_confirm_button


# -----------------------------------------------------------
# PAGE HEADER
# -----------------------------------------------------------
st.markdown("""
<h1 class="section-title">🔎 Search & Rank Restaurants</h1>
<p class="subtext">
Use this tool to look up any restaurant in our dataset, review details, 
and add your personal rating to your ranking list.
</p>
<hr>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# INITIALIZE RANKED LIST IF NEEDED
# -----------------------------------------------------------
if "ranked_restaurants" not in st.session_state:
    st.session_state["ranked_restaurants"] = []


# -----------------------------------------------------------
# RESTAURANT SEARCH INPUT
# -----------------------------------------------------------
restaurants = load_yelp_restaurants()

st.markdown("""
<h3 class="section-subtitle">📍 Find a Restaurant</h3>
<p class="subtext">Start typing the name of a place — the search bar autocompletes with matches.</p>
""", unsafe_allow_html=True)

options = {
    f"{row['name']} ({row['city']}, {row['state']})": i
    for i, row in restaurants.iterrows()
}

user_choice = st.selectbox(
    "Select a restaurant to view details:",
    list(options.keys()),
    help="Begin typing a restaurant name to filter the list."
)

idx = options[user_choice]


# -----------------------------------------------------------
# RESTAURANT DISPLAY CARD
# -----------------------------------------------------------
st.markdown("""
<h3 class="section-subtitle">🍽️ Restaurant Details</h3>
<p class="subtext">Here’s the restaurant you selected. Review it below and decide whether you want to rank it.</p>
""", unsafe_allow_html=True)

restaurant_card(restaurants, idx)


# -----------------------------------------------------------
# RATING SLIDER + CONFIRM BUTTON
# -----------------------------------------------------------
st.markdown("""
<br>
<h3 class="section-subtitle">⭐ Rate This Restaurant</h3>
<p class="subtext">Choose a score from 1–10 and add it to your personal ranking list.</p>
""", unsafe_allow_html=True)

show_slider_and_confirm_button(idx)
