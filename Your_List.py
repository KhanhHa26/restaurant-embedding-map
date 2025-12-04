# source venv_name/bin/activate 
# venv/bin/python -m streamlit run Your_List.py

import streamlit as st
from PIL import Image
from utils.data_loader import load_yelp_restaurants
from utils.slider import show_slider_and_confirm_button

# -------------------------
# Set up + titles 
# -------------------------

im = Image.open("restaurant_icon.png")
st.set_page_config(
    page_title="TasteMap",
    page_icon=im,
    layout="wide"
)

st.markdown("""
<h1 style='text-align:center; margin-bottom: -10px;'>TasteMap - Discover & Rank Restaurants</h1>
<p style='text-align:center; font-size:18px; color:#666'>
Visualize restaurants, search similar places, and rank your favorites.
</p>
""", unsafe_allow_html=True)

st.divider()
st.write("👈 Use the sidebar to navigate between pages.")

# Load restaurants ONCE
restaurants = load_yelp_restaurants()

# Initialize ranking list
if "ranked_restaurants" not in st.session_state:
    st.session_state["ranked_restaurants"] = []

# -------------------------
# User's Ranked Restaurant List 
# -------------------------
st.header("🍽️ Your Ranked Restaurants")

if st.session_state["ranked_restaurants"]:
    to_remove = None
    for idx, score in st.session_state["ranked_restaurants"]:
        st.markdown(f"""
        <div style="padding:10px; border-radius:8px; background:#fafafa; 
                    margin-bottom:10px; border:1px solid #eee; color:#111;">
            <h4 style="margin:0;">{restaurants.iloc[idx]['name']}</h4>
            ⭐ {restaurants.iloc[idx]['stars']} — {restaurants.iloc[idx]['city']}
            <p><strong>Your Score:</strong> {score}</p>
        </div>
        """, unsafe_allow_html=True)

        # Button alignment: side-by-side
        col1, col2 = st.columns([0.15, 0.15])

        with col1:
            if st.button("🗑️ Delete", key=f"delete_{idx}"):
                st.session_state["ranked_restaurants"].remove((idx, score))
                st.rerun()

        with col2:
            if st.button("✏️ Edit", key=f"edit_{idx}"):
                show_slider_and_confirm_button(idx)
else:
    st.info("You haven't ranked any restaurants yet. Try searching or viewing similar restaurants to begin!")
