# 🎬 Movie Rating Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange.svg)](https://scikit-learn.org/)

---

## 📖 Description

This project predicts **movie ratings** (on a scale of 1–10) from movie attributes using five machine learning regression algorithms. It covers the full ML pipeline — synthetic dataset generation, exploratory data analysis, preprocessing, model training, evaluation, comparison and visualization — and ends with an **interactive predictor** where you enter a movie's details and get its predicted rating.

---

## 📊 Dataset

`movie_dataset.csv` contains **2,000 synthetic movie records** created by `generate_dataset.py` (fixed random seed, so it is reproducible). There are no missing values.

| # | Feature | Description | Type | Range |
|---|---------|-------------|------|-------|
| 1 | `Genre` | Genre of the movie (Action, Comedy, Drama, …) | Categorical | 10 genres |
| 2 | `Director_Popularity` | Popularity score of the director | Numerical | 1.0 – 10.0 |
| 3 | `Cast_Popularity` | Popularity score of the cast | Numerical | 1.2 – 10.0 |
| 4 | `Budget_Millions` | Production budget in millions of USD | Numerical | 1 – 300 |
| 5 | `Runtime_Minutes` | Duration of the movie in minutes | Numerical | 80 – 181 |
| 6 | `Year` | Release year of the movie | Numerical | 2000 – 2025 |
| 7 | `Votes` | Number of user votes received | Numerical | 1,000 – 156,416 |
| 8 | `Revenue_Millions` | Box office revenue in millions of USD | Numerical | 0.5 – 1,298.7 |

`Movie_Name` is also in the file but is dropped before training, since a title carries no predictive information.

### 🎯 Target Variable
- **`Rating`** — Movie rating on a scale of **1.0 to 10.0** (mean 5.45)

### How the data is generated
Each rating is a weighted combination of genre baseline, director popularity, cast popularity and (log-scaled) budget, **plus random noise**, rescaled to 1–10. Revenue and votes are then derived from budget, popularity and rating. Because of the built-in noise, no model can predict ratings perfectly.

---

## 🤖 Algorithms Used

| # | Algorithm | Description |
|---|-----------|-------------|
| 1 | **Linear Regression** | A baseline linear approach that models the rating as a weighted sum of features. |
| 2 | **Decision Tree Regressor** | Splits the data into branches based on feature thresholds. |
| 3 | **Random Forest Regressor** | An ensemble of 100 decision trees whose predictions are averaged to reduce overfitting. |
| 4 | **Gradient Boosting Regressor** | Builds 100 trees sequentially, each one correcting the errors of the previous ones. |
| 5 | **Support Vector Regressor (SVR)** | A kernel-based (RBF) method that fits a function keeping most errors within a margin. |

All models use `random_state=42` (where applicable) and are trained on the same 80/20 train-test split. The model with the highest R² on the test set is selected as the best model.

---

## 🔧 Preprocessing

1. Handle missing values — median for numeric columns, mode for categorical columns
2. Label-encode `Genre`
3. Drop `Movie_Name`
4. Scale features with `StandardScaler`
5. Train-test split — 80% train (1,600 movies) / 20% test (400 movies)

---

## 📁 Project Structure

```
ML mini project/
│
├── 📄 README.md                       # Project documentation (this file)
├── 📄 requirements.txt                # Python dependencies
├── 📄 how to run.txt                  # Quick run instructions
├── 📄 generate_dataset.py             # Generates the synthetic movie dataset
├── 📄 movie_dataset.csv               # The dataset (2,000 movies)
├── 📄 movie_rating_prediction.py      # Main ML pipeline: EDA, training, evaluation, plots
├── 📄 predict_movie.py                # Interactive predictor for new movies
├── 📄 best_model.pkl                  # Saved best model (Gradient Boosting)
├── 📄 model_artifacts.pkl             # All trained models + scaler + encoder + results
├── 📄 Movie_Rating_Prediction.pptx    # Project presentation
│
└── 📂 plots/                          # Visualizations
    ├── model_comparison.png
    ├── actual_vs_predicted.png
    ├── feature_importance.png
    ├── correlation_heatmap.png
    ├── genre_distribution.png
    └── rating_distribution.png
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1: Generate the Dataset (optional)
`movie_dataset.csv` is already included. To regenerate it:
```bash
python generate_dataset.py
```

### Step 2: Train and Evaluate the Models
```bash
python movie_rating_prediction.py
```
> This will:
> - Load and explore the dataset
> - Preprocess the data
> - Train and evaluate all 5 regression models
> - Compare them and pick the best one
> - Generate visualizations in the `plots/` folder
> - Save the best model as `best_model.pkl` and everything needed for prediction as `model_artifacts.pkl`

### Step 3: Predict a Movie's Rating
```bash
python predict_movie.py
```
Enter the movie name, choose a genre and type in the 7 numeric details. The script shows the prediction from every model and the final **predicted rating** from the best model:

```
  Gradient Boosting     8.16 / 10   (R² = 0.347)  <-- best model
  SVR                   6.89 / 10   (R² = 0.325)
  Random Forest         7.68 / 10   (R² = 0.307)
  Linear Regression     8.49 / 10   (R² = 0.300)
  Decision Tree         9.20 / 10   (R² = -0.375)
------------------------------------------------------------
  PREDICTED RATING: 8.2 / 10  (by Gradient Boosting)
```

---

## 📏 Evaluation Metrics

| Metric | Full Name | Description |
|--------|-----------|-------------|
| **MAE** | Mean Absolute Error | Average of the absolute differences between predicted and actual values. Lower is better. |
| **MSE** | Mean Squared Error | Average of the squared differences between predicted and actual values. Penalizes large errors more heavily. |
| **RMSE** | Root Mean Squared Error | Square root of MSE, expressed in the same unit as the rating. Lower is better. |
| **R² Score** | Coefficient of Determination | Proportion of rating variance explained by the model (1 = perfect fit). Higher is better. |

---

## 📈 Results

### Model Comparison (test set, 400 movies)

| Model | MAE | MSE | RMSE | R² Score |
|-------|-----|-----|------|----------|
| **Gradient Boosting** ⭐ | **0.9123** | **1.3167** | **1.1475** | **0.3466** |
| SVR | 0.9279 | 1.3607 | 1.1665 | 0.3248 |
| Random Forest | 0.9468 | 1.3973 | 1.1821 | 0.3066 |
| Linear Regression | 0.9369 | 1.4106 | 1.1877 | 0.3000 |
| Decision Tree | 1.3337 | 2.7713 | 1.6647 | -0.3752 |

**Gradient Boosting** is the best model: its predictions are off by about 0.9 rating points on average. The single Decision Tree overfits the noise in the data and performs worse than simply predicting the average (negative R²).

### Most Important Features (Random Forest)
Revenue (0.29), Director Popularity (0.16), Cast Popularity (0.15) and Budget (0.10) matter most; Runtime, Year and Genre contribute little.

### Visualizations

The `plots/` directory contains:
- 📊 **Model Comparison** — R² scores of all models
- 📊 **Actual vs Predicted** — Prediction accuracy of the best model
- 📊 **Feature Importance** — Most influential features
- 📊 **Correlation Heatmap** — Feature correlations
- 📊 **Genre Distribution** — Number of movies per genre
- 📊 **Rating Distribution** — Distribution of the target variable

---

## ⚠️ Limitations & Future Scope

- The dataset is synthetic; training on real data (IMDb / TMDB) would make the results more meaningful.
- Revenue and votes are only known after release, so a pre-release model should exclude them.
- Models use default hyperparameters; adding cross-validation and GridSearchCV tuning, and trying XGBoost / LightGBM, are natural next steps.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Core programming language |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Machine learning algorithms & evaluation |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Data manipulation & analysis |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical computations |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logoColor=white) | Data visualization |
| ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat&logoColor=white) | Statistical data visualization |
| joblib | Saving and loading trained models |

---

## 👤 Author

**Vatsal Chaudhari** — ID: 23IT439
Subject: Machine Learning — Mini Project

---

## 📝 License

This project is for educational purposes as part of a Machine Learning mini project.

---

<p align="center">⭐ If you found this project helpful, give it a star! ⭐</p>
