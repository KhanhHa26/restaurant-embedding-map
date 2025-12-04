import pandas as pd
from kagglehub import KaggleDatasetAdapter
import kagglehub
import streamlit as st

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

def combine_text(row):
    fields = []
    for col in ["name", "categories", "attributes", "city"]:
        if pd.notna(row[col]):
            fields.append(str(row[col]))
    return " ".join(fields)

