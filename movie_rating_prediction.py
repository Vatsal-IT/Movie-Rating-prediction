# ============================================================================
# Movie Rating Prediction - ML Mini Project
# ============================================================================
# This project predicts movie ratings using various machine learning regression
# models. It demonstrates a complete ML pipeline: data loading, exploration,
# preprocessing, model training, evaluation, comparison, and visualization.
# ============================================================================

# =========================== IMPORTS ========================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plot style for consistent, attractive visualizations
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    plt.style.use('ggplot')  # Fallback style if seaborn style not available

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)


# ============================================================================
# SECTION 1: DATA LOADING & EXPLORATION
# ============================================================================
print("=" * 60)
print("       SECTION 1: DATA LOADING & EXPLORATION")
print("=" * 60)

# Load the dataset from CSV file
df = pd.read_csv('movie_dataset.csv')

# Display basic dataset information
print("\n Dataset Shape (rows, columns):", df.shape)

print("\n Dataset Info:")
print("-" * 40)
df.info()

print("\n Statistical Summary:")
print("-" * 40)
print(df.describe())

print("\n Missing Values per Column:")
print("-" * 40)
missing = df.isnull().sum()
print(missing)
print(f"\nTotal missing values: {missing.sum()}")

print("\n First 5 Rows of the Dataset:")
print("-" * 40)
print(df.head())

print("\n Data loading & exploration complete!\n")


# ============================================================================
# SECTION 2: DATA PREPROCESSING
# ============================================================================
print("=" * 60)
print("       SECTION 2: DATA PREPROCESSING")
print("=" * 60)

# --- Step 2.1: Handle Missing Values (if any) ---
# Fill missing numeric values with the median (robust to outliers)
# Fill missing categorical values with the mode (most frequent value)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Filled missing values in '{col}' with median: {median_val}")

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"  Filled missing values in '{col}' with mode: {mode_val}")

print("\n Missing values handled.")

# --- Step 2.2: Label Encode the 'Genre' Column ---
# Convert categorical genre labels into numeric values so ML models can use them
label_encoder = LabelEncoder()
if 'Genre' in df.columns:
    df['Genre'] = label_encoder.fit_transform(df['Genre'])
    print(f"\n  Label Encoded 'Genre' column.")
    print(f"   Genre classes: {list(label_encoder.classes_)}")
    print(f"   Encoded values: {list(range(len(label_encoder.classes_)))}")

# --- Step 2.3: Drop 'Movie_Name' Column ---
# Movie names are unique identifiers and don't contribute to prediction
if 'Movie_Name' in df.columns:
    df.drop('Movie_Name', axis=1, inplace=True)
    print("\n  Dropped 'Movie_Name' column (not useful for prediction).")

print("\n Columns after preprocessing:", list(df.columns))

# --- Step 2.4: Define Features (X) and Target (y) ---
# Separate the dataset into input features and the target variable
X = df.drop('Rating', axis=1)  # All columns except 'Rating'
y = df['Rating']               # Target: the rating we want to predict

print(f"\n Feature columns: {list(X.columns)}")
print(f" Target column: Rating")
print(f"   Features shape: {X.shape}")
print(f"   Target shape: {y.shape}")

# --- Step 2.5: Feature Scaling using StandardScaler ---
# Standardize features to have mean=0 and std=1 for better model performance
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)
print("\n  Feature scaling applied (StandardScaler).")

# --- Step 2.6: Train-Test Split ---
# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42  # For reproducibility
)

print(f"\n  Train-Test Split (80/20):")
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Testing set:  {X_test.shape[0]} samples")

print("\n Data preprocessing complete!\n")


# ============================================================================
# SECTION 3: MODEL TRAINING & EVALUATION
# ============================================================================
print("=" * 60)
print("       SECTION 3: MODEL TRAINING & EVALUATION")
print("=" * 60)

# Define the models to train and compare
# Each model offers different strengths for regression tasks
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR()
}

# Dictionary to store evaluation results for each model
results = {
    'Model': [],
    'MAE': [],
    'MSE': [],
    'RMSE': [],
    'R² Score': []
}

# Dictionary to store trained models and their predictions
trained_models = {}
model_predictions = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n Training: {name}...")

    # Train the model on training data
    model.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)      # Mean Absolute Error
    mse = mean_squared_error(y_test, y_pred)        # Mean Squared Error
    rmse = np.sqrt(mse)                              # Root Mean Squared Error
    r2 = r2_score(y_test, y_pred)                    # R-squared Score

    # Store results
    results['Model'].append(name)
    results['MAE'].append(round(mae, 4))
    results['MSE'].append(round(mse, 4))
    results['RMSE'].append(round(rmse, 4))
    results['R² Score'].append(round(r2, 4))

    # Store the trained model and its predictions for later use
    trained_models[name] = model
    model_predictions[name] = y_pred

    # Print individual model results
    print(f"   MAE:      {mae:.4f}")
    print(f"   MSE:      {mse:.4f}")
    print(f"   RMSE:     {rmse:.4f}")
    print(f"   R² Score: {r2:.4f}")

print("\n All models trained and evaluated!\n")


# ============================================================================
# SECTION 4: RESULTS COMPARISON
# ============================================================================
print("=" * 60)
print("       SECTION 4: RESULTS COMPARISON")
print("=" * 60)

# Create a DataFrame from the results dictionary for easy comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R² Score', ascending=False).reset_index(drop=True)

print("\n Model Performance Comparison:")
print("-" * 70)
print(results_df.to_string(index=False))
print("-" * 70)

# Identify the best model based on highest R² Score
best_model_name = results_df.iloc[0]['Model']
best_r2 = results_df.iloc[0]['R² Score']

print(f"\n Best Model: {best_model_name}")
print(f"   R² Score:   {best_r2}")
print(f"   MAE:        {results_df.iloc[0]['MAE']}")
print(f"   RMSE:       {results_df.iloc[0]['RMSE']}")

# Get the best model object and its predictions
best_model = trained_models[best_model_name]
best_predictions = model_predictions[best_model_name]

print("\n Results comparison complete!\n")


# ============================================================================
# SECTION 5: VISUALIZATIONS
# ============================================================================
print("=" * 60)
print("       SECTION 5: VISUALIZATIONS")
print("=" * 60)

# --- Plot 1: Model Comparison (R² Scores) ---
print("\n Generating: Model R² Score Comparison...")
fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette('viridis', len(results_df))
bars = ax.bar(results_df['Model'], results_df['R² Score'], color=colors, edgecolor='black')

# Add value labels on top of each bar
for bar, score in zip(bars, results_df['R² Score']):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
            f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_title('Model Comparison - R² Scores', fontsize=16, fontweight='bold')
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_ylim(0, max(results_df['R² Score']) * 1.15)  # Add some headroom
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('plots/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: plots/model_comparison.png")

# --- Plot 2: Actual vs Predicted (Best Model) ---
print("\n Generating: Actual vs Predicted (Best Model)...")
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, best_predictions, alpha=0.6, color='steelblue',
           edgecolors='navy', s=50, label='Predictions')

# Plot the perfect prediction line (y = x)
min_val = min(y_test.min(), best_predictions.min())
max_val = max(y_test.max(), best_predictions.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

ax.set_title(f'Actual vs Predicted Ratings\n(Best Model: {best_model_name})',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Actual Rating', fontsize=12)
ax.set_ylabel('Predicted Rating', fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('plots/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: plots/actual_vs_predicted.png")

# --- Plot 3: Feature Importance (Random Forest) ---
print("\n Generating: Feature Importance (Random Forest)...")
rf_model = trained_models['Random Forest']
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
feature_importances.plot(kind='barh', color=sns.color_palette('magma', len(feature_importances)),
                         edgecolor='black', ax=ax)

# Add value labels at the end of each bar
for i, (val, name) in enumerate(zip(feature_importances, feature_importances.index)):
    ax.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=10)

ax.set_title('Feature Importance (Random Forest)', fontsize=16, fontweight='bold')
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: plots/feature_importance.png")

# --- Plot 4: Correlation Heatmap ---
print("\n Generating: Correlation Heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))
# Compute correlation matrix for all numeric features (including encoded Genre)
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle

sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, ax=ax,
            cbar_kws={'shrink': 0.8})

ax.set_title('Correlation Heatmap of Numeric Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: plots/correlation_heatmap.png")

# --- Plot 5: Genre Distribution ---
print("\n Generating: Genre Distribution...")
# Reload original data to get original genre names for the plot
df_original = pd.read_csv('movie_dataset.csv')

fig, ax = plt.subplots(figsize=(12, 6))
if 'Genre' in df_original.columns:
    genre_counts = df_original['Genre'].value_counts()
    colors = sns.color_palette('Set2', len(genre_counts))
    genre_counts.plot(kind='bar', color=colors, edgecolor='black', ax=ax)

    # Add count labels on top of bars
    for i, (val, name) in enumerate(zip(genre_counts, genre_counts.index)):
        ax.text(i, val + 0.5, str(val), ha='center', va='bottom', fontweight='bold', fontsize=9)

ax.set_title('Distribution of Movies by Genre', fontsize=16, fontweight='bold')
ax.set_xlabel('Genre', fontsize=12)
ax.set_ylabel('Number of Movies', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/genre_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: plots/genre_distribution.png")

# --- Plot 6: Rating Distribution ---
print("\n Generating: Rating Distribution...")
fig, ax = plt.subplots(figsize=(10, 6))

# Histogram with KDE (Kernel Density Estimation) overlay
ax.hist(df['Rating'], bins=20, color='skyblue', edgecolor='black', alpha=0.7, density=True, label='Histogram')
df['Rating'].plot(kind='kde', ax=ax, color='darkblue', linewidth=2, label='KDE')

ax.axvline(df['Rating'].mean(), color='red', linestyle='--', linewidth=1.5,
           label=f'Mean: {df["Rating"].mean():.2f}')
ax.axvline(df['Rating'].median(), color='green', linestyle='--', linewidth=1.5,
           label=f'Median: {df["Rating"].median():.2f}')

ax.set_title('Distribution of Movie Ratings', fontsize=16, fontweight='bold')
ax.set_xlabel('Rating', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('plots/rating_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: plots/rating_distribution.png")

print("\n All visualizations saved to 'plots/' directory!\n")


# ============================================================================
# SECTION 6: SAVE BEST MODEL
# ============================================================================
print("=" * 60)
print("       SECTION 6: SAVE BEST MODEL")
print("=" * 60)

# Save the best performing model to disk using joblib
# This allows the model to be loaded and used later without retraining
model_filename = 'best_model.pkl'
joblib.dump(best_model, model_filename)
print(f"\n Best model ({best_model_name}) saved to '{model_filename}'")

# Verify the saved model by loading it back
loaded_model = joblib.load(model_filename)
verify_pred = loaded_model.predict(X_test[:1])
print(f"   Verification - Prediction on first test sample: {verify_pred[0]:.4f}")
print(f"   Actual value: {y_test.iloc[0]:.4f}")

# Save everything needed to predict on NEW movies (used by predict_movie.py):
# the trained models plus the fitted scaler and genre encoder
artifacts = {
    'best_model_name': best_model_name,
    'models': trained_models,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'feature_columns': list(X.columns),
    'results': results_df,
}
joblib.dump(artifacts, 'model_artifacts.pkl')
print(" Models, scaler & encoder saved to 'model_artifacts.pkl'")

print("\n Model saved successfully!\n")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 60)
print("       PROJECT COMPLETE - SUMMARY")
print("=" * 60)
print(f"""
 Files Generated:
   - plots/model_comparison.png      : R² score comparison bar chart
   - plots/actual_vs_predicted.png   : Actual vs predicted scatter plot
   - plots/feature_importance.png    : Random Forest feature importance
   - plots/correlation_heatmap.png   : Feature correlation heatmap
   - plots/genre_distribution.png    : Genre distribution bar chart
   - plots/rating_distribution.png   : Rating distribution histogram
   - best_model.pkl                  : Saved best model

 Best Model: {best_model_name} (R² = {best_r2})
 Total Models Evaluated: {len(models)}
 Features Used: {list(X.columns)}
 Dataset Size: {df.shape[0]} rows × {df.shape[1]} columns
""")
print("=" * 60)
print("       Thank you! Project execution finished.")
print("=" * 60)
