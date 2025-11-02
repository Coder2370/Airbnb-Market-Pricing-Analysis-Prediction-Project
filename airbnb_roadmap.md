# Airbnb Pricing Prediction - Complete Project Roadmap & Code Repository

## Quick Start Guide

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv airbnb_env
source airbnb_env/bin/activate  # On Windows: airbnb_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
# Option A: Inside Airbnb (Recommended - Free, regularly updated)
# Visit https://insideairbnb.com/get-the-data/
# Download listings.csv for your city of choice

# Option B: Kaggle (Multiple options available)
# Visit https://www.kaggle.com/datasets/
# Popular options:
# - New York Airbnb Open Data 2024
# - Chicago Airbnb Open Data
# - Your City name + Airbnb

# After downloading, place in: data/raw/listings.csv
```

### 3. Run Full Pipeline
```bash
# Execute complete analysis
python airbnb_pipeline.py

# Or run Jupyter notebooks sequentially
jupyter notebook notebooks/01_data_loading_exploration.ipynb
jupyter notebook notebooks/02_eda_visualization.ipynb
# ... continue with remaining notebooks
```

---

## Project Dependencies

**File:** `requirements.txt`

```
pandas==2.0.0
numpy==1.24.0
matplotlib==3.7.0
seaborn==0.12.0
scikit-learn==1.3.0
xgboost==2.0.0
lightgbm==4.0.0
plotly==5.16.0
jupyter==1.0.0
notebook==6.5.0
scipy==1.10.0
statsmodels==0.14.0
```

**Installation:**
```bash
pip install -r requirements.txt
```

---

## Detailed Notebook Structure

### Notebook 1: Data Loading & Initial Exploration
**File:** `01_data_loading_exploration.ipynb`

**Objectives:**
- Load and inspect raw dataset
- Display basic information (shape, columns, dtypes)
- Initial statistical summary
- Identify missing values and data quality issues

**Key Cells:**
```python
# Load data
df = pd.read_csv('data/raw/listings.csv')
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Identify target and features
print(df['price'].describe())
print(df.columns.tolist())
```

**Output:**
- Data overview report
- Missing values heatmap
- Statistical summary table

---

### Notebook 2: Exploratory Data Analysis
**File:** `02_eda_visualization.ipynb`

**Objectives:**
- Understand price distribution
- Analyze relationships between features and price
- Identify outliers and anomalies
- Geographic and neighborhood analysis

**Key Visualizations:**
1. Price distribution (histogram, KDE, Q-Q plot)
2. Price by room type (bar chart, violin plot)
3. Price by neighborhood (top 20 neighborhoods)
4. Correlations with price (heatmap, bar chart)
5. Geographic map of listings with price overlay
6. Price trends by availability
7. Review patterns vs price

**Code Example:**
```python
# Price distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Histogram
axes[0,0].hist(df['price'], bins=50, edgecolor='black')
axes[0,0].set_title('Price Distribution')

# Box plot by room type
df.boxplot(column='price', by='room_type', ax=axes[0,1])

# Top neighborhoods
top_neighborhoods = df.groupby('neighbourhood')['price'].mean().nlargest(15)
top_neighborhoods.plot(kind='barh', ax=axes[1,0])
axes[1,0].set_title('Top 15 Neighborhoods by Average Price')

# Correlations
numeric_df = df.select_dtypes(include=[np.number])
correlations = numeric_df.corr()['price'].sort_values(ascending=False)
correlations.plot(kind='barh', ax=axes[1,1])
axes[1,1].set_title('Feature Correlations with Price')

plt.tight_layout()
plt.savefig('reports/eda_summary.png', dpi=300, bbox_inches='tight')
```

---

### Notebook 3: Data Cleaning & Preprocessing
**File:** `03_data_cleaning_preprocessing.ipynb`

**Objectives:**
- Clean messy data (handle missing values, outliers)
- Standardize data types
- Remove duplicates
- Prepare for modeling

**Key Steps:**
```python
# 1. Missing values
print(df.isnull().sum())
# Fill or drop based on strategy
df['bedrooms'].fillna(df['bedrooms'].median(), inplace=True)
df['reviews_per_month'].fillna(0, inplace=True)

# 2. Outliers
Q1 = df['price'].quantile(0.01)
Q3 = df['price'].quantile(0.99)
df = df[(df['price'] >= Q1) & (df['price'] <= Q3)]

# 3. Data types
df['price'] = df['price'].astype(float)
df['host_identity_verified'] = df['host_identity_verified'].astype(int)

# 4. Duplicates
df = df.drop_duplicates()

# 5. Save cleaned data
df.to_csv('data/processed/listings_cleaned.csv', index=False)
```

---

### Notebook 4: Feature Engineering
**File:** `04_feature_engineering.ipynb`

**Objectives:**
- Create meaningful features from raw data
- Extract insights from text fields
- Generate interaction features
- Encode categorical variables

**Feature Engineering Strategies:**

**A) Categorical Encoding:**
```python
# One-hot encoding for room type
room_type_dummies = pd.get_dummies(df['room_type'], prefix='room_type')
df = pd.concat([df, room_type_dummies], axis=1)

# Label encoding for ordinal data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['neighbourhood_encoded'] = le.fit_transform(df['neighbourhood'])
```

**B) Numerical Transformations:**
```python
# Create derived features
df['price_per_person'] = df['price'] / (df['accommodates'] + 1)
df['bedrooms_per_guest'] = df['bedrooms'] / (df['accommodates'] + 1)
df['availability_pct'] = df['availability_365'] / 365

# Polynomial features
df['accommodates_squared'] = df['accommodates'] ** 2
df['log_price'] = np.log1p(df['price'])
```

**C) Text-based Features (Amenities):**
```python
# Extract amenities from text
df['has_wifi'] = df['amenities'].str.contains('Wifi', case=False, na=False).astype(int)
df['has_kitchen'] = df['amenities'].str.contains('Kitchen', case=False, na=False).astype(int)
df['has_parking'] = df['amenities'].str.contains('Free parking', case=False, na=False).astype(int)
df['amenity_count'] = df['amenities'].str.count(',') + 1
```

**D) Time-based Features:**
```python
# Convert to datetime
df['first_review'] = pd.to_datetime(df['first_review'])
df['last_review'] = pd.to_datetime(df['last_review'])

# Days since last review
df['days_since_review'] = (pd.Timestamp.now() - df['last_review']).dt.days
df['years_hosting'] = (pd.Timestamp.now() - df['first_review']).dt.days / 365
```

**E) Geographic Features:**
```python
# Location clusters
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=42)
df['location_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])

# Distance to city center (example for New York)
df['distance_to_center'] = np.sqrt(
    (df['latitude'] - 40.7128)**2 + 
    (df['longitude'] - (-74.0060))**2
)
```

**Output:**
- Feature engineering report
- Engineered dataset with 50+ features
- Feature importance baseline

---

### Notebook 5: Modeling & Evaluation
**File:** `05_modeling_evaluation.ipynb`

**Objectives:**
- Build multiple regression models
- Compare performance metrics
- Optimize hyperparameters
- Select best model

**Model Building:**
```python
from sklearn.model_selection import train_test_split, cross_val_score

# Prepare data
X = df.drop(['price', 'id', 'name', 'description'], axis=1)
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

# XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)

# Evaluate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print(f"RF R²: {r2_score(y_test, rf_pred):.4f}")
print(f"RF RMSE: ${np.sqrt(mean_squared_error(y_test, rf_pred)):.2f}")
print(f"XGB R²: {r2_score(y_test, xgb_pred):.4f}")
print(f"XGB RMSE: ${np.sqrt(mean_squared_error(y_test, xgb_pred)):.2f}")
```

**Hyperparameter Tuning:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(
    XGBRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV R²: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
best_pred = best_model.predict(X_test_scaled)
print(f"Test R²: {r2_score(y_test, best_pred):.4f}")
```

---

## Key Analysis Results

### Typical Findings (Example Output)

**Model Performance Comparison:**
| Model | Train R² | Test R² | RMSE | MAE |
|-------|----------|---------|------|-----|
| Linear Regression | 0.52 | 0.48 | $85 | $62 |
| Decision Tree | 0.78 | 0.58 | $75 | $55 |
| Random Forest | 0.82 | 0.65 | $68 | $48 |
| Gradient Boosting | 0.81 | 0.66 | $65 | $45 |
| **XGBoost (Best)** | **0.80** | **0.67** | **$62** | **$42** |

**Top 10 Feature Importance (XGBoost):**
1. room_type_Entire_home (0.185)
2. accommodates (0.156)
3. neighbourhood_encoded (0.142)
4. bedrooms (0.128)
5. minimum_nights (0.095)
6. has_wifi (0.087)
7. price_per_person (0.076)
8. availability_365 (0.064)
9. reviews_per_month (0.055)
10. host_listings_count (0.042)

---

## Production Implementation

### Creating a Prediction API

**File:** `src/api.py` (Optional Flask implementation)

```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict price for given property features"""
    data = request.json
    
    # Convert to DataFrame
    features = pd.DataFrame([data])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    
    return jsonify({
        'predicted_price': float(prediction),
        'currency': 'USD',
        'model': 'XGBoost'
    })

if __name__ == '__main__':
    app.run(debug=True)
```

**Usage:**
```bash
# Start API
python src/api.py

# Make prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "accommodates": 4,
    "bedrooms": 2,
    "bathrooms": 1.5,
    "room_type_Entire_home": 1,
    "neighbourhood_encoded": 15,
    "minimum_nights": 30,
    "availability_365": 200,
    "reviews_per_month": 2.5,
    "has_wifi": 1
  }'
```

---

## GitHub Repository Structure

```
airbnb-pricing-prediction/
├── README.md                          # Project overview
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
│
├── data/
│   ├── raw/
│   │   ├── .gitkeep
│   │   └── README.md                # Data source documentation
│   ├── processed/
│   │   └── .gitkeep
│   └── external/
│       └── .gitkeep
│
├── notebooks/
│   ├── 01_data_loading_exploration.ipynb
│   ├── 02_eda_visualization.ipynb
│   ├── 03_data_cleaning_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_modeling_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── utils.py
│   ├── api.py                       # Flask API (optional)
│   └── config.py
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── encoders.pkl
│
├── reports/
│   ├── findings.md
│   ├── technical_report.pdf
│   ├── visualizations/
│   │   ├── eda_summary.png
│   │   ├── price_distribution.png
│   │   ├── correlations.png
│   │   ├── model_comparison.png
│   │   ├── feature_importance.png
│   │   ├── predictions_analysis.png
│   │   └── residuals_plot.png
│   └── presentation/
│       └── airbnb_pricing_analysis.pptx
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   └── test_models.py
│
└── docs/
    ├── project_guide.md
    ├── architecture.md
    ├── api_documentation.md
    └── methodology.md
```

---

## Writing the Technical Report

**File:** `reports/technical_report.md` (8-12 pages)

### Report Structure
1. **Executive Summary** (1 page)
2. **Introduction & Business Context** (1-2 pages)
3. **Methodology** (2 pages)
   - Data Collection
   - Feature Engineering
   - Model Selection
4. **Results** (2-3 pages)
   - Model Performance Comparison
   - Feature Importance Analysis
   - Key Findings
5. **Insights & Recommendations** (2 pages)
   - Pricing Drivers
   - Recommendations for Hosts
   - Market Insights
6. **Conclusion & Future Work** (1 page)
7. **Appendix** (References, Additional Visualizations)

---

## Portfolio Presentation Tips

### Key Talking Points
- "Built end-to-end ML pipeline predicting Airbnb prices with 67% R² score"
- "Engineered 50+ features including amenity extraction and geographic clustering"
- "Compared 5 regression models using cross-validation; XGBoost achieved best performance"
- "Identified location and room type as top price drivers (18.5% and 15.6% importance)"
- "Optimized hyperparameters reducing RMSE by 12% through GridSearchCV"

### Live Demo Ideas
- Show prediction for new listing (enter features, get price estimate)
- Interactive dashboard with price prediction by neighborhood
- Model performance visualization comparing predicted vs actual
- Feature importance breakdown for specific predictions

### Interview Preparation
- Be ready to explain why you chose each model
- Discuss trade-offs between accuracy and interpretability
- Explain feature engineering decisions
- Talk about handling of missing values and outliers
- Discuss cross-validation and avoiding overfitting

---

## Common Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Skewed price distribution | Log transformation, quantile regression |
| Missing values | Forward fill for time-series, median for numerical, mode for categorical |
| Categorical variables with many levels | Target encoding, frequency encoding, or clustering |
| Multicollinearity | Feature selection, correlation analysis, PCA |
| Outliers in price | Quantile-based filtering, IQR method |
| Imbalanced features | Standardization, normalization |
| Model overfitting | Cross-validation, regularization, hyperparameter tuning |
| Class imbalance (if using classification) | SMOTE, class weights, stratified splitting |

---

## Performance Benchmarks

**Expected Results by Model:**

**Linear Regression:** R² ≈ 0.48-0.52 (baseline)
**Decision Tree:** R² ≈ 0.58-0.62
**Random Forest:** R² ≈ 0.63-0.68
**Gradient Boosting:** R² ≈ 0.64-0.69
**XGBoost:** R² ≈ 0.65-0.70 (typically best)

**Best Practice:** If your results are significantly worse, investigate:
- Data quality issues
- Missing critical features
- Feature engineering gaps
- Improper data scaling
- Hyperparameter tuning needed

---
