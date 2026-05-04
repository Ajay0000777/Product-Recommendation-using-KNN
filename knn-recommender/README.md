# ProximaRec — KNN Product Recommendation System

A production-ready product recommendation engine built with **Flask** and **scikit-learn**,
using the **K-Nearest Neighbors (KNN)** algorithm to find products that match your preferences
in a multi-dimensional feature space.

---

## File Structure

```
knn-recommender/
├── app.py                  # Flask application + KNN logic
├── requirements.txt        # Python dependencies
├── README.md
├── templates/
│   └── index.html          # Jinja2 HTML template
└── static/
    └── style.css           # Stylesheet
```

---

## Setup & Running

### 1. Clone / copy the files

```bash
mkdir knn-recommender && cd knn-recommender
# place app.py, requirements.txt, templates/index.html, static/style.css
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## How the KNN Algorithm Works Here

### Overview

KNN is a *lazy learning* algorithm — it memorizes the training data and, at query time,
finds the K points in that dataset that are "closest" to a new query point.
There is no training phase; all computation happens at inference.

### Feature Vector

Each product is represented as a 5-dimensional vector:

```
[price, rating, category_id, popularity, discount_pct]
```

Example: Sony WH-1000XM5 → `[349.99, 4.8, 1, 95, 10]`

### Normalization (MinMaxScaler)

Raw features have very different scales (price can be $5–$2000; rating is 1–5).
Without normalization, the price dimension would dominate the distance calculation.

MinMaxScaler maps every feature to [0, 1]:

```
x_scaled = (x - x_min) / (x_max - x_min)
```

The same scaler fitted on the product matrix is used to transform the user's
query vector before the KNN search.

### Distance Metric

Euclidean distance between two n-dimensional points A and B:

```
d(A, B) = sqrt( Σ (A_i - B_i)² )
```

Lower distance = more similar products.

### Similarity Score

Distance is converted to an intuitive 0–100% similarity score:

```
similarity = max(0, (1 - distance) × 100)
```

### Two-Phase Filtering

1. **KNN phase**: retrieve K+1 nearest neighbors purely by geometry.
2. **Hard filter phase**: remove products that strictly violate user constraints
   (e.g. price > max_budget). This ensures KNN handles fuzzy "closeness"
   while hard constraints remain respected.

If strict filtering leaves too few results (<2), the category filter is relaxed
automatically so the user always gets useful recommendations.

### API Endpoint

A JSON API is also available for programmatic access:

```bash
curl -X POST http://127.0.0.1:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"max_price": 200, "min_rating": 4.5, "category": "Electronics", "n_results": 3}'
```

---

## Customization

| What to change | Where |
|---|---|
| Add/remove products | `PRODUCTS` list in `app.py` |
| Change K value | `NearestNeighbors(n_neighbors=...)` in `app.py` |
| Change distance metric | `metric="euclidean"` → `"manhattan"`, `"cosine"`, etc. |
| Add new features | Extend feature vector in `build_feature_matrix()` |
| Change port | `app.run(port=5000)` at the bottom of `app.py` |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| Flask | ≥ 2.3 | Web framework, routing, templating |
| scikit-learn | ≥ 1.3 | KNN (`NearestNeighbors`), `MinMaxScaler` |
| NumPy | ≥ 1.24 | Numeric array operations |
