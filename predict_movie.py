# ============================================================================
# Movie Rating Predictor - Interactive
# ============================================================================
# Enter a movie's name and details, and the trained ML models predict its
# rating. Run 'python movie_rating_prediction.py' first to train the models.
# ============================================================================

import os
import sys
import warnings

import joblib
import pandas as pd

warnings.filterwarnings('ignore')

ARTIFACTS_FILE = 'model_artifacts.pkl'

# Prompt text and valid range for each numeric feature (based on the dataset)
NUMERIC_FIELDS = {
    'Director_Popularity': ('Director popularity (1-10)', 1, 10, float),
    'Cast_Popularity':     ('Cast popularity (1-10)', 1, 10, float),
    'Budget_Millions':     ('Budget in millions $ (e.g. 50)', 0.1, 1000, float),
    'Runtime_Minutes':     ('Runtime in minutes (e.g. 120)', 30, 300, int),
    'Year':                ('Release year (e.g. 2023)', 1900, 2100, int),
    'Votes':               ('Number of votes (e.g. 50000)', 0, 10_000_000, int),
    'Revenue_Millions':    ('Revenue in millions $ (e.g. 120)', 0, 5000, float),
}


# =========================== INPUT HELPERS ==================================
def ask_number(prompt, low, high, cast):
    """Keep asking until the user enters a valid number in [low, high]."""
    while True:
        raw = input(f"  {prompt}: ").strip()
        try:
            value = cast(float(raw))
        except ValueError:
            print("    Please enter a number.")
            continue
        if low <= value <= high:
            return value
        print(f"    Please enter a value between {low} and {high}.")


def ask_genre(genres):
    """Let the user pick a genre by number or by name."""
    print("  Genres:")
    for i, g in enumerate(genres, 1):
        print(f"    {i:2d}. {g}")
    lookup = {g.lower(): g for g in genres}
    while True:
        raw = input("  Choose genre (number or name): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(genres):
            return genres[int(raw) - 1]
        if raw.lower() in lookup:
            return lookup[raw.lower()]
        print("    Invalid genre, try again.")


# =========================== PREDICTION =====================================
def predict(artifacts, movie):
    """Encode, scale and run every model on one movie. Returns {model: rating}."""
    row = pd.DataFrame([movie])
    row['Genre'] = artifacts['label_encoder'].transform(row['Genre'])
    row = row[artifacts['feature_columns']]
    row_scaled = pd.DataFrame(artifacts['scaler'].transform(row),
                              columns=artifacts['feature_columns'])

    # Ratings in the dataset are on a 1-10 scale, so clip to that range
    return {name: float(min(max(model.predict(row_scaled)[0], 1), 10))
            for name, model in artifacts['models'].items()}


def main():
    if not os.path.exists(ARTIFACTS_FILE):
        print(f" '{ARTIFACTS_FILE}' not found.")
        print(" Run 'python movie_rating_prediction.py' first to train the models.")
        sys.exit(1)

    artifacts = joblib.load(ARTIFACTS_FILE)
    best_name = artifacts['best_model_name']
    genres = list(artifacts['label_encoder'].classes_)
    r2_scores = dict(zip(artifacts['results']['Model'], artifacts['results']['R² Score']))

    print("=" * 60)
    print("            MOVIE RATING PREDICTOR")
    print("=" * 60)

    while True:
        print()
        name = input("  Movie name: ").strip() or "Untitled Movie"
        movie = {'Genre': ask_genre(genres)}
        for col, (prompt, low, high, cast) in NUMERIC_FIELDS.items():
            movie[col] = ask_number(prompt, low, high, cast)

        predictions = predict(artifacts, movie)

        print("\n" + "-" * 60)
        print(f"  Predictions for '{name}' ({movie['Genre']})")
        print("-" * 60)
        for model_name, rating in sorted(predictions.items(),
                                         key=lambda kv: -r2_scores.get(kv[0], 0)):
            tag = "  <-- best model" if model_name == best_name else ""
            print(f"  {model_name:<20} {rating:5.2f} / 10   (R² = {r2_scores[model_name]:.3f}){tag}")
        print("-" * 60)
        print(f"  PREDICTED RATING: {predictions[best_name]:.1f} / 10  (by {best_name})")
        print("-" * 60)

        again = input("\n  Predict another movie? (y/n): ").strip().lower()
        if again != 'y':
            print("\n  Goodbye!")
            break


if __name__ == '__main__':
    main()
