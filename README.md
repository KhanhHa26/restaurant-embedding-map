# 🍽️ Restaurant Embedding Explorer

An interactive Streamlit app that visualizes Yelp restaurants in a 2D embedding space and recommends similar restaurants using text embeddings and nearest-neighbor search.

---

## 🌟 Overview

This project turns restaurant metadata (name, categories, attributes, city) from the **Yelp Open Dataset** into vector embeddings using **Sentence Transformers**. These embeddings are reduced to 2D using **UMAP** for visualization and used with cosine similarity to find semantically similar restaurants.

<img width="1131" height="463" alt="Screenshot 2025-12-01 at 2 47 42 AM" src="https://github.com/user-attachments/assets/f447f436-2a2d-4c64-b766-b85a0fa34e08" />

---

## ✨ Features

- **Interactive Embedding Map**  
  Visualizes thousands of restaurants using UMAP, colored by category.

- **Semantic Similarity Search**  
  Enter a restaurant name and retrieve the top 10 most similar restaurants.

- **Fast Lookup**  
  Uses cached embeddings and a cosine-distance-based nearest-neighbor model.

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
````

### 2. Run the app

```bash
streamlit run app.py
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
