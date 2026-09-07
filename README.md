# California Housing Price Prediction

An end-to-end machine-learning case study for predicting median housing values in California using **data cleaning, geographic feature engineering and XGBoost regression**.

The project focuses on turning a small tabular dataset into a richer modeling dataset through domain-driven features, validating model performance, and reducing the final feature set for better interpretability.

## Highlights

- **20,640** original California districts from the 1990 census dataset
- Missing-value analysis, consistency checks and capped-value handling
- Geographic enrichment using city proximity and a gravity-style attractiveness score
- Ratio, interaction, binning and log-transformed features
- XGBoost regression with cross-validation and randomized hyperparameter search
- Reduction from roughly **37 model features to 15 selected features**
- Exploratory holdout performance of **R² = 0.8338** and **RMSE ≈ $40.3k**

## Project workflow

```text
Raw housing.csv
      |
      v
Exploration & cleaning
      |
      v
Geographic + business feature engineering
      |
      v
Model dataset
      |
      v
XGBoost + cross-validation + tuning
      |
      v
Feature analysis & housing-price predictions
```

## Results

| Metric | Result |
| --- | ---: |
| Cross-validation R² | 0.8317 |
| Holdout R² | **0.8338** |
| Holdout RMSE | **$40,273** |
| Final selected features | 15 |

The strongest predictors in the exploratory model are related to **household income**, followed by **location**, **distance to major California cities**, **ocean proximity**, and engineered geographic/economic interactions.

![California housing prices map](assets/california_price_map.png)

### Feature-count experiment

The project also explores how predictive performance changes as lower-importance variables are removed. The performance curve becomes nearly flat after roughly 15–20 features, motivating a more compact model.

![R2 versus number of features](assets/r2_vs_features.png)

## Feature engineering

The second notebook creates features designed to capture both geographic and socioeconomic effects.

### Geographic features

- Haversine distances to major California cities
- Distance to the nearest selected city
- North/south and east/west regional brackets
- A gravity-style score combining city population and distance
- Ocean-proximity encoding

### Structural and socioeconomic features

- Rooms per household
- Bedroom ratio
- Population per household
- Income per person
- Income and housing-age interactions
- Geographic × income interactions
- Log transformations for strongly skewed variables

## Repository structure

```text
.
├── assets/
│   ├── california_price_map.png
│   └── r2_vs_features.png
├── data/
│   ├── raw/
│   ├── preprocessed/
│   └── README.md
├── models/
│   └── README.md
├── notebooks/
│   ├── 01_data_exploration_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling_xgboost.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

## Tech stack

**Python · Pandas · NumPy · Scikit-learn · XGBoost · GeoPandas · Contextily · Matplotlib · Seaborn · Jupyter**

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/california-housing-price-prediction.git
cd california-housing-price-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the California Housing Prices dataset from Kaggle and place:

```text
housing.csv
```

inside:

```text
data/raw/
```

See [`data/README.md`](data/README.md) for details.

### 5. Run the notebooks in order

```text
notebooks/01_data_exploration_cleaning.ipynb
notebooks/02_feature_engineering.ipynb
notebooks/03_modeling_xgboost.ipynb
```

## Validation note

This is an **applied machine-learning / portfolio project**, not a production benchmark. During exploratory feature reduction, the holdout set is used to visualize performance as the number of features changes, and the final feature shortlist is also informed by a model fitted on the full dataset. As a result, the reported holdout metric should be treated as an exploratory estimate rather than a perfectly untouched final-test score.

A stricter production-grade version would perform **feature selection and hyperparameter tuning entirely inside the training/CV pipeline**, then evaluate only once on a final untouched test set (or use nested cross-validation).

## Limitations and next steps

- The dataset represents California housing in **1990**, so the model is not suitable for current-market valuation.
- The original target contains an artificial upper cap, which requires careful treatment.
- Important real-estate drivers such as school quality, crime, transport access, lot size and temporal trends are unavailable.
- Geographic features could be enriched with external census or OpenStreetMap data.
- A future iteration could compare XGBoost with LightGBM/CatBoost and add SHAP-based explanations.

## Dataset

**California Housing Prices** — Kaggle / Cam Nugent  
https://www.kaggle.com/datasets/camnugent/california-housing-prices

The original dataset is not redistributed in this repository.

## Author

**Jules Courné**  
Data Analyst / Data Engineer · Python · SQL · Machine Learning
