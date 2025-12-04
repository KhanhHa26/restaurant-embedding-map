import streamlit as st
import plotly.express as px
from utils.data_loader import load_yelp_restaurants, combine_text
from utils.embeddings import load_embeddings
from utils.umap_reducer import get_umap_projection

# -----------------------------------------------------------
# PAGE HEADER
# -----------------------------------------------------------
st.markdown("""
<h1 class='section-title'>📍 Visual Restaurant Map</h1>
<p class='subtext'>
This map places restaurants in a 2D space based on their text embeddings.  
Points closer together represent restaurants that are more similar in description, cuisine, and attributes.
</p>
<hr>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# LOAD & PREP DATA
# -----------------------------------------------------------
restaurants = load_yelp_restaurants()
restaurants["text"] = restaurants.apply(combine_text, axis=1)

# Embeddings + UMAP
embeddings = load_embeddings(restaurants["text"])
X_umap = get_umap_projection(embeddings)

# Extract primary category
restaurants["main_category"] = restaurants["categories"].apply(
    lambda x: x.split(",")[0].strip() if isinstance(x, str) else "Other"
)


# -----------------------------------------------------------
# INTERACTIVE VISUALIZATION
# -----------------------------------------------------------
st.markdown("""
<h3 class='section-subtitle'>🌐 Explore the Map</h3>
<p class='subtext'>
Hover over any point to view restaurant details.  
Use this map to understand patterns in cuisine types and similarity relationships.
</p>
""", unsafe_allow_html=True)


@st.cache_resource
def build_map():
    fig = px.scatter(
        restaurants,
        x=X_umap[:, 0],
        y=X_umap[:, 1],
        color="main_category",
        hover_data=["name", "stars", "city"],
        opacity=0.75,
    )
    fig.update_layout(
        height=650,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",  
        legend_title_text="Main Category",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


# Optional: Wrap the plot in a soft card
st.markdown("<div class='card'>", unsafe_allow_html=True)
fig = build_map()
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------
# SMALL INFO BOX / FOOTNOTE
# -----------------------------------------------------------
st.markdown("""
<p class='muted' style='margin-top: 20px; font-size:14px;'>
✨ This visualization uses <strong>SentenceTransformer embeddings</strong> + 
<strong>UMAP dimensionality reduction</strong> to place restaurants in a similarity space.
</p>
""", unsafe_allow_html=True)
