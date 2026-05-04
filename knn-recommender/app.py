"""
KNN Product Recommendation System
Flask application with scikit-learn KNN implementation.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import json

app = Flask(__name__)

# ─────────────────────────────────────────────
#  SAMPLE PRODUCT DATASET
#  Features: price (USD), rating (1-5), category_id,
#            popularity (0-100), discount_pct (0-100)
# ─────────────────────────────────────────────
PRODUCTS = [
    {"id": 1,  "name": "Sony WH-1000XM5 Headphones",  "category": "Electronics",  "price": 349.99, "rating": 4.8, "popularity": 95, "discount": 10, "image_icon": "🎧", "description": "Industry-leading noise cancellation with 30-hr battery."},
    {"id": 2,  "name": "Kindle Paperwhite",            "category": "Electronics",  "price": 139.99, "rating": 4.7, "popularity": 88, "discount": 15, "image_icon": "📖", "description": "Adjustable warm light, waterproof, weeks of battery."},
    {"id": 3,  "name": "Nike Air Max 270",             "category": "Footwear",     "price": 150.00, "rating": 4.5, "popularity": 90, "discount": 5,  "image_icon": "👟", "description": "Max Air cushioning with a modern silhouette."},
    {"id": 4,  "name": "Instant Pot Duo 7-in-1",      "category": "Kitchen",      "price": 89.99,  "rating": 4.7, "popularity": 92, "discount": 20, "image_icon": "🍲", "description": "Pressure cooker, slow cooker, rice cooker and more."},
    {"id": 5,  "name": "Levi's 501 Original Jeans",   "category": "Clothing",     "price": 69.50,  "rating": 4.4, "popularity": 85, "discount": 0,  "image_icon": "👖", "description": "Iconic straight-fit jeans, timeless since 1873."},
    {"id": 6,  "name": "Apple AirPods Pro (2nd Gen)", "category": "Electronics",  "price": 249.00, "rating": 4.8, "popularity": 97, "discount": 8,  "image_icon": "🎵", "description": "Adaptive Transparency, Personalized Spatial Audio."},
    {"id": 7,  "name": "Yoga Mat Premium Cork",       "category": "Sports",       "price": 45.00,  "rating": 4.6, "popularity": 78, "discount": 10, "image_icon": "🧘", "description": "Natural cork surface, anti-slip, eco-friendly."},
    {"id": 8,  "name": "Dyson V15 Detect",            "category": "Home",         "price": 699.99, "rating": 4.7, "popularity": 82, "discount": 5,  "image_icon": "🧹", "description": "Laser dust detection, HEPA filtration, 60-min runtime."},
    {"id": 9,  "name": "The Alchemist – Hardcover",   "category": "Books",        "price": 14.99,  "rating": 4.6, "popularity": 94, "discount": 0,  "image_icon": "📚", "description": "Paulo Coelho's timeless tale of following your dream."},
    {"id": 10, "name": "Whey Protein Isolate 5lb",    "category": "Sports",       "price": 59.99,  "rating": 4.5, "popularity": 87, "discount": 12, "image_icon": "💪", "description": "25g protein per serving, low carb, fast absorption."},
    {"id": 11, "name": "Nespresso Vertuo Pop",        "category": "Kitchen",      "price": 119.00, "rating": 4.6, "popularity": 80, "discount": 18, "image_icon": "☕", "description": "Centrifusion tech, 5 cup sizes, milk frother included."},
    {"id": 12, "name": "Patagonia Nano Puff Jacket",  "category": "Clothing",     "price": 229.00, "rating": 4.7, "popularity": 76, "discount": 0,  "image_icon": "🧥", "description": "Lightweight, windproof, 60g PrimaLoft insulation."},
    {"id": 13, "name": "Logitech MX Master 3S",       "category": "Electronics",  "price": 99.99,  "rating": 4.8, "popularity": 89, "discount": 7,  "image_icon": "🖱️", "description": "8K DPI sensor, MagSpeed scroll, silent clicks."},
    {"id": 14, "name": "Adidas Ultraboost 23",        "category": "Footwear",     "price": 189.99, "rating": 4.6, "popularity": 88, "discount": 10, "image_icon": "🏃", "description": "BOOST midsole, PRIMEKNIT+ upper, Continental rubber."},
    {"id": 15, "name": "Vitamix 5200 Blender",        "category": "Kitchen",      "price": 449.95, "rating": 4.8, "popularity": 75, "discount": 0,  "image_icon": "🥤", "description": "Aircraft-grade stainless blades, 7-year warranty."},
    {"id": 16, "name": "Moleskine Classic Notebook",  "category": "Books",        "price": 19.95,  "rating": 4.5, "popularity": 86, "discount": 5,  "image_icon": "📓", "description": "Hard cover, ruled pages, elastic closure, ribbon."},
    {"id": 17, "name": "TRX Suspension Trainer",      "category": "Sports",       "price": 149.95, "rating": 4.6, "popularity": 72, "discount": 8,  "image_icon": "🏋️", "description": "Full-body workout, door anchor included, military-grade."},
    {"id": 18, "name": "Roomba j7+ Robot Vacuum",     "category": "Home",         "price": 599.99, "rating": 4.5, "popularity": 79, "discount": 15, "image_icon": "🤖", "description": "Obstacle avoidance, self-emptying base, smart mapping."},
]

# ─────────────────────────────────────────────
#  CATEGORY → NUMERIC ENCODING
# ─────────────────────────────────────────────
CATEGORY_MAP = {
    "Electronics": 1, "Footwear": 2, "Kitchen": 3,
    "Clothing": 4,    "Sports": 5,   "Home": 6, "Books": 7,
}

def build_feature_matrix(products):
    """
    Convert product list into a numeric feature matrix.
    Features: [price, rating, category_id, popularity, discount_pct]
    """
    matrix = []
    for p in products:
        matrix.append([
            p["price"],
            p["rating"],
            CATEGORY_MAP.get(p["category"], 0),
            p["popularity"],
            p["discount"],
        ])
    return np.array(matrix, dtype=float)


# Pre-build and cache the KNN model at startup
_scaler = MinMaxScaler()
_feature_matrix = build_feature_matrix(PRODUCTS)
_scaled_matrix = _scaler.fit_transform(_feature_matrix)

# k=6 so we can return top-5 excluding the query point itself when it matches
_knn_model = NearestNeighbors(n_neighbors=6, metric="euclidean", algorithm="auto")
_knn_model.fit(_scaled_matrix)


def get_recommendations(user_prefs: dict, n: int = 5):
    """
    Given user preferences dict, return top-N recommended products.

    KNN workflow:
    1. Build a query vector from user preferences.
    2. Scale it with the same scaler used on training data.
    3. Find K nearest neighbors in feature space.
    4. Return those products with distance-based similarity scores.

    Parameters
    ----------
    user_prefs : dict  Keys: max_price, min_rating, category, min_popularity, min_discount
    n          : int   Number of results to return (default 5)
    """

    # --- Build query vector ---
    category_id = CATEGORY_MAP.get(user_prefs.get("category", ""), 0)
    # If no category selected (0), use midpoint so it doesn't skew results
    if category_id == 0:
        category_id = 4.0

    query = np.array([[
        float(user_prefs.get("max_price", 300)),
        float(user_prefs.get("min_rating", 4.0)),
        float(category_id),
        float(user_prefs.get("min_popularity", 70)),
        float(user_prefs.get("min_discount", 0)),
    ]])

    # --- Scale using the pre-fit scaler ---
    query_scaled = _scaler.transform(query)

    # --- KNN query ---
    k = min(n + 1, len(PRODUCTS))
    distances, indices = _knn_model.kneighbors(query_scaled, n_neighbors=k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        product = PRODUCTS[idx].copy()

        # Convert Euclidean distance → similarity percentage (0–100)
        # Smaller distance = more similar
        similarity = round(max(0, (1 - dist) * 100), 1)
        product["similarity"] = similarity
        product["distance"] = round(float(dist), 4)

        # Apply hard filters AFTER KNN (KNN finds geometrically similar;
        # filters remove anything that strictly violates user constraints)
        max_price = float(user_prefs.get("max_price", 99999))
        min_rating = float(user_prefs.get("min_rating", 0))
        pref_category = user_prefs.get("category", "")

        if product["price"] > max_price:
            continue
        if product["rating"] < min_rating:
            continue
        if pref_category and product["category"] != pref_category:
            continue  # only filter by category when explicitly chosen

        results.append(product)

        if len(results) >= n:
            break

    # If strict filters left us with too few, relax category filter and retry
    if len(results) < 2:
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            product = PRODUCTS[idx].copy()
            similarity = round(max(0, (1 - dist) * 100), 1)
            product["similarity"] = similarity
            product["distance"] = round(float(dist), 4)
            results.append(product)
            if len(results) >= n:
                break

    return results


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    categories = list(CATEGORY_MAP.keys())
    return render_template("index.html", categories=categories)


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        prefs = {
            "max_price":      request.form.get("max_price", 500),
            "min_rating":     request.form.get("min_rating", 4.0),
            "category":       request.form.get("category", ""),
            "min_popularity": request.form.get("min_popularity", 70),
            "min_discount":   request.form.get("min_discount", 0),
            "n_results":      int(request.form.get("n_results", 5)),
        }

        n = max(1, min(int(prefs["n_results"]), 10))
        recommendations = get_recommendations(prefs, n=n)

        categories = list(CATEGORY_MAP.keys())
        return render_template(
            "index.html",
            categories=categories,
            recommendations=recommendations,
            prefs=prefs,
            total_products=len(PRODUCTS),
        )

    except Exception as e:
        categories = list(CATEGORY_MAP.keys())
        return render_template(
            "index.html",
            categories=categories,
            error=f"Something went wrong: {str(e)}",
        ), 500


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """JSON API endpoint for programmatic access."""
    data = request.get_json(force=True) or {}
    recommendations = get_recommendations(data, n=int(data.get("n_results", 5)))
    return jsonify({"status": "ok", "count": len(recommendations), "results": recommendations})


if __name__ == "__main__":
    print("🚀  KNN Recommender running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
