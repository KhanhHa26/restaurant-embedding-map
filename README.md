## 🍽️ TasteMap — Restaurant Discovery & Ranking Explorer

An interactive Streamlit application that lets you visualize, compare, and rank restaurants using semantic text embeddings and nearest-neighbor search.
Built on the Yelp Open Dataset, TasteMap combines data visualization, similarity search, and personalized scoring into one intuitive interface.

---

## 🌟 Overview

TasteMap transforms restaurant metadata (name, categories, attributes, city) into vector embeddings using Sentence Transformers. These embeddings power:
- A 2D UMAP visualization of the restaurant landscape
<img width="1131" height="463" alt="Screenshot 2025-12-03 at 10 08 01 PM" src="https://github.com/user-attachments/assets/b2c80663-3ce2-4d95-b006-8d129a9dd165" />

- A semantic similarity engine to find related restaurants
<img width="1131" height="463" alt="Screenshot 2025-12-03 at 10 04 16 PM" src="https://github.com/user-attachments/assets/f0807df4-73e8-4b6d-91e2-e1fc62afb7c5" />

- A personal ranking tool where users score and curate their own list
<img width="1131" height="463" alt="Screenshot 2025-12-03 at 10 05 08 PM" src="https://github.com/user-attachments/assets/964b92ef-86d6-4a48-b286-dbd23003799f" />


---

## ✨ Features

**📍 Visual Restaurant Map**
- UMAP reduces 384-dimensional embeddings into an interactive 2D scatter plot.
- Points are color-coded by main restaurant category (e.g., Chinese, Café, Pizza).
- Hover to view names, cities, and ratings.

**🔍 Search & Rank**
- Search any restaurant in the dataset.
- Rate it on a customizable 1–10 scale and add it to your personal, persistent ranking list.

**✨ Similar Restaurant Explorer**
- Select a restaurant and discover the top nearest neighbors by cosine similarity.
- Score and add any similar restaurant to your ranking list.

**📜 Your Ranked Restaurant List**
- A dedicated page showing your curated selections.
- Edit or remove entries easily.

**⚡ Optimized Performance**
- Embeddings are cached locally for instant reloads.
- UMAP and NearestNeighbors are precomputed for fast similarity lookup.

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
````

### 2. Run the app

```bash
streamlit run Your_List.py
```

The app will launch at: [http://localhost:8501](http://localhost:8501)

---

## 🧠 How It Works

* Restaurants are transformed into text strings combining name + categories + attributes.
* `all-MiniLM-L6-v2` generates 384-dimensional embeddings.
* Embeddings are reduced to 2D via **UMAP** for visualization.
* Similarity search uses cosine distance from **NearestNeighbors**.

---

## 📦 Data Source

This project uses the **Yelp Open Dataset**, loaded through KaggleHub:

```
yelp_academic_dataset_business.json
```

Filtered to include only entries with "Restaurants" in their categories.

---

## 📜 License

This project is for educational and research purposes and uses publicly available Yelp data.
