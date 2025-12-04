
import streamlit as st

def show_slider_and_confirm_button(idx):
    score = st.slider("Rate this restaurant", 1.0, 10.0, 5.0, 0.1, key=f"slider_{idx}")

    if st.button("Confirm",key={f"confirm_button_{idx}"}):
        st.session_state["ranked_restaurants"].append((idx, score))
        st.success("Added to ranking list!")