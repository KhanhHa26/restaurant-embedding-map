import streamlit as st

def restaurant_card(df, idx):
    row = df.iloc[idx]
    st.markdown(f"""
        <div style="padding:12px; border-radius:10px; background:#fafafa;
             margin-bottom:10px; border:1px solid #eee; color:#111;">
            <h4>{row['name']}</h4>
            ⭐ {row['stars']} — {row['city']}, {row['state']}<br>
            <strong>Categories:</strong> {row['categories']}
        </div>
    """, unsafe_allow_html=True)
