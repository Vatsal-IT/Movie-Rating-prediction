"""
generate_dataset.py
====================
Generates a realistic synthetic movie dataset (~2000 rows) for ML mini project.
The dataset includes features that realistically influence a movie's rating,
with controlled correlations and noise to mimic real-world data.

Output: movie_dataset.csv
"""

import numpy as np
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================
np.random.seed(42)  # For reproducibility
NUM_MOVIES = 2000

# ----------------------------------------------------------------------------
# Movie Name Generation
# ----------------------------------------------------------------------------
# We combine adjectives, nouns, and optional suffixes to create
# realistic-sounding movie titles (e.g., "The Dark Horizon", "Eternal Shadows")

adjectives = [
    "Dark", "Lost", "Eternal", "Silent", "Broken", "Hidden", "Final",
    "Crimson", "Golden", "Savage", "Frozen", "Rising", "Fallen", "Twisted",
    "Burning", "Shattered", "Iron", "Mystic", "Hollow", "Raging",
    "Distant", "Wicked", "Fading", "Sacred", "Phantom", "Electric",
    "Scarlet", "Silver", "Bitter", "Velvet", "Neon", "Cursed", "Fearless",
    "Last", "Wild", "Bright", "Midnight", "Deadly", "Ancient", "Ruthless",
]

nouns = [
    "Horizon", "Shadows", "Legacy", "Storm", "Revenge", "Kingdom",
    "Redemption", "Secrets", "Empire", "Justice", "Destiny", "Fury",
    "Reckoning", "Odyssey", "Silence", "Echoes", "Flames", "Ashes",
    "Venom", "Paradise", "Nightmare", "Fortress", "Chronicle", "Inferno",
    "Blade", "Dawn", "Requiem", "Awakening", "Prophecy", "Eclipse",
    "Labyrinth", "Dominion", "Abyss", "Mirage", "Phantom", "Pursuit",
    "Vendetta", "Sentinel", "Cascade", "Fractured",
]

prefixes = ["The", "A", "Beyond", "Into the", "Return of the", ""]

def generate_movie_names(n):
    """Generate n unique-ish movie names by combining prefixes, adjectives, and nouns."""
    names = []
    for _ in range(n):
        prefix = np.random.choice(prefixes)
        adj = np.random.choice(adjectives)
        noun = np.random.choice(nouns)
        if prefix:
            name = f"{prefix} {adj} {noun}"
        else:
            name = f"{adj} {noun}"
        # Append a small random number suffix to avoid exact duplicates
        names.append(name)

    # Ensure uniqueness by appending a number to duplicates
    seen = {}
    unique_names = []
    for name in names:
        if name in seen:
            seen[name] += 1
            unique_names.append(f"{name} {seen[name]}")
        else:
            seen[name] = 0
            unique_names.append(name)
    return unique_names


# ============================================================================
# Generate Core Features
# ============================================================================
print("Generating synthetic movie dataset...")

movie_names = generate_movie_names(NUM_MOVIES)

# Genre distribution — some genres are more common than others
genres = [
    "Action", "Comedy", "Drama", "Horror", "Sci-Fi",
    "Romance", "Thriller", "Animation", "Documentary", "Adventure",
]
genre_weights = [0.15, 0.14, 0.16, 0.10, 0.09, 0.08, 0.10, 0.06, 0.05, 0.07]
genre_col = np.random.choice(genres, size=NUM_MOVIES, p=genre_weights)

# Director Popularity: float 1.0–10.0
# Most directors are mid-range; a few are very popular (right-skewed distribution)
director_popularity = np.clip(
    np.random.beta(2, 5, size=NUM_MOVIES) * 10 + 1, 1.0, 10.0
)
director_popularity = np.round(director_popularity, 1)

# Cast Popularity: float 1.0–10.0
# Similar skewed distribution — big-name casts are rarer
cast_popularity = np.clip(
    np.random.beta(2.5, 4, size=NUM_MOVIES) * 10 + 1, 1.0, 10.0
)
cast_popularity = np.round(cast_popularity, 1)

# Budget (Millions): float 1–300
# Log-normal distribution — most movies are low-to-mid budget, few blockbusters
budget = np.clip(
    np.random.lognormal(mean=3.5, sigma=0.9, size=NUM_MOVIES), 1, 300
)
budget = np.round(budget, 2)

# Runtime (Minutes): int 80–200
# Normal distribution centered around 115 minutes
runtime = np.clip(
    np.random.normal(loc=115, scale=20, size=NUM_MOVIES), 80, 200
).astype(int)

# Year: int 2000–2025
# Slightly more recent movies (linearly increasing probability)
year_weights = np.arange(1, 27, dtype=float)  # 26 years: 2000–2025
year_weights /= year_weights.sum()
year = np.random.choice(range(2000, 2026), size=NUM_MOVIES, p=year_weights)


# ============================================================================
# Generate Rating (Target Variable)
# ============================================================================
# Rating is realistically influenced by multiple features with added noise:
#   - Higher director popularity  → slightly higher rating
#   - Higher cast popularity      → slightly higher rating
#   - Higher budget               → diminishing returns (log-scaled)
#   - Genre baseline              → some genres tend to rate higher (e.g., Drama)
#   - Noise                       → significant random component (real ratings vary!)

# Genre-based baseline (some genres tend to have higher average ratings)
genre_baseline = {
    "Action": 5.8, "Comedy": 5.9, "Drama": 6.5, "Horror": 5.2,
    "Sci-Fi": 6.1, "Romance": 5.7, "Thriller": 6.2, "Animation": 6.4,
    "Documentary": 6.8, "Adventure": 6.0,
}
genre_base = np.array([genre_baseline[g] for g in genre_col])

# Build rating from components
rating = (
    genre_base * 0.30                                           # Genre influence
    + (director_popularity / 10) * 2.0                          # Director influence
    + (cast_popularity / 10) * 1.5                              # Cast influence
    + np.log1p(budget) / np.log1p(300) * 1.5                    # Budget (diminishing)
    + np.random.normal(0, 0.8, size=NUM_MOVIES)                 # Random noise
)

# Normalize to 1.0–10.0 range
rating = (rating - rating.min()) / (rating.max() - rating.min()) * 9 + 1
rating = np.clip(np.round(rating, 1), 1.0, 10.0)


# ============================================================================
# Generate Revenue (Correlated with Budget and Rating)
# ============================================================================
# Revenue depends on budget (bigger budget → bigger marketing → more revenue),
# rating (better movies earn more via word-of-mouth), and random luck.

revenue = (
    budget * np.random.uniform(0.5, 3.0, size=NUM_MOVIES)      # Budget multiplier
    * (rating / 5.0)                                            # Rating boost
    + np.random.normal(0, 10, size=NUM_MOVIES)                  # Noise
)
revenue = np.clip(np.round(revenue, 2), 0.5, None)  # Minimum $0.5M revenue


# ============================================================================
# Generate Votes (Correlated with Popularity and Budget)
# ============================================================================
# More popular casts/directors and higher budgets attract more voter attention

votes = (
    budget * 200                                                 # Budget draws attention
    + director_popularity * 5000                                 # Director fan base
    + cast_popularity * 8000                                     # Cast fan base
    + np.random.normal(0, 15000, size=NUM_MOVIES)                # Noise
)
votes = np.clip(votes.astype(int), 1000, 500000)


# ============================================================================
# Assemble DataFrame
# ============================================================================
df = pd.DataFrame({
    "Movie_Name": movie_names,
    "Genre": genre_col,
    "Director_Popularity": director_popularity,
    "Cast_Popularity": cast_popularity,
    "Budget_Millions": budget,
    "Runtime_Minutes": runtime,
    "Year": year,
    "Votes": votes,
    "Revenue_Millions": revenue,
    "Rating": rating,
})

# ============================================================================
# Save to CSV
# ============================================================================
output_path = "movie_dataset.csv"
df.to_csv(output_path, index=False)
print(f"Dataset saved to '{output_path}'\n")

# ============================================================================
# Print Summary
# ============================================================================
print("=" * 60)
print("           DATASET SUMMARY")
print("=" * 60)
print(f"Total movies: {len(df)}")
print(f"Columns: {list(df.columns)}")
print()

print("--- Genre Distribution ---")
print(df["Genre"].value_counts().to_string())
print()

print("--- Numerical Feature Statistics ---")
print(df.describe().round(2).to_string())
print()

print("--- First 5 Rows ---")
print(df.head().to_string())
print()

print("--- Correlation with Rating (target) ---")
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_cols].corr()["Rating"].drop("Rating").sort_values(ascending=False)
print(correlations.round(3).to_string())
print()
print("Dataset generation complete!")
