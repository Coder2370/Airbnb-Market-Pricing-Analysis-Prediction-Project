"""
Airbnb Pricing Prediction - Complete Implementation
Main analysis and modeling pipeline
Author: Data Analyst
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: DATA LOADING & EXPLORATION
# ============================================================================

def load_data(filepath):
    """Load and display basic information about dataset"""
    df = pd.read_csv(filepath)
    print("Dataset Shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nBasic statistics:")
    print(df.describe())
    return df

# ============================================================================
# PART 2: DATA CLEANING & PREPROCESSING
# ============================================================================

def clean_data(df):
    """Clean and prepare data for analysis"""
    df_clean = df.copy()
    
    # Remove rows with missing target variable
    if 'price' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['price'])
    
    # Handle price formatting (remove $ and commas if needed)
    if df_clean['price'].dtype == 'object':
        df_clean['price'] = df_clean['price'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Remove outliers in price (keep reasonable range)
    Q1 = df_clean['price'].quantile(0.01)
    Q3 = df_clean['price'].quantile(0.99)
    df_clean = df_clean[(df_clean['price'] >= Q1) & (df_clean['price'] <= Q3)]
    
    # Fill missing values in categorical columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown', inplace=True)
    
    # Fill missing values in numerical columns
    numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        if col != 'price':  # Don't fill price column
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    print(f"Cleaned data shape: {df_clean.shape}")
    return df_clean

# ============================================================================
# PART 3: EXPLORATORY DATA ANALYSIS
# ============================================================================

def exploratory_analysis(df):
    """Perform comprehensive EDA"""
    
    # Price distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histogram
    axes[0, 0].hist(df['price'], bins=50, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Price Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Price ($)')
    
    # Box plot
    axes[0, 1].boxplot(df['price'])
    axes[0, 1].set_title('Price Box Plot', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Price ($)')
    
    # Room type comparison
    if 'room_type' in df.columns:
        room_prices = df.groupby('room_type')['price'].mean().sort_values(ascending=False)
        axes[1, 0].bar(room_prices.index, room_prices.values, color='coral', edgecolor='black')
        axes[1, 0].set_title('Average Price by Room Type', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Average Price ($)')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Correlations (numerical features only)
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) > 1:
        correlations = df[numerical_cols].corr()['price'].sort_values(ascending=False)
        axes[1, 1].barh(range(min(10, len(correlations))), correlations.values[:min(10, len(correlations))])
        axes[1, 1].set_yticks(range(min(10, len(correlations))))
        axes[1, 1].set_yticklabels(correlations.index[:min(10, len(correlations))])
        axes[1, 1].set_title('Top 10 Correlations with Price', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Correlation Coefficient')
    
    plt.tight_layout()
    plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
    print("EDA visualization saved as 'eda_analysis.png'")
    plt.show()

# ============================================================================
# PART 4: FEATURE ENGINEERING
# ============================================================================

def feature_engineering(df):
    """Create new features and engineer existing ones"""
    df_features = df.copy()
    
    # 1. Encode categorical variables
    categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()
    le_dict = {}
    
    for col in categorical_cols:
        if col in df_features.columns:
            le = LabelEncoder()
            df_features[col + '_encoded'] = le.fit_transform(df_features[col])
            le_dict[col] = le
    
    # 2. Create interaction features
    if 'accommodates' in df_features.columns and 'bedrooms' in df_features.columns:
        df_features['price_per_person'] = df_features['price'] / (df_features['accommodates'] + 1)
    
    if 'number_of_reviews' in df_features.columns:
        df_features['review_score_per_month'] = df_features['number_of_reviews'] / (df_features['reviews_per_month'] + 1)
    
    # 3. Polynomial features for key numerical columns
    numerical_features = ['accommodates', 'bedrooms', 'minimum_nights']
    for feature in numerical_features:
        if feature in df_features.columns:
            df_features[f'{feature}_squared'] = df_features[feature] ** 2
            df_features[f'{feature}_log'] = np.log1p(df_features[feature])
    
    # 4. Binning features
    if 'price' in df_features.columns:
        df_features['price_category'] = pd.cut(df_features['price'], 
                                               bins=[0, 100, 200, 300, float('inf')],
                                               labels=['Budget', 'Mid-range', 'Premium', 'Luxury'])
    
    print(f"Features after engineering: {df_features.shape[1]} (original: {df.shape[1]})")
    return df_features, le_dict

# ============================================================================
# PART 5: MODEL BUILDING & COMPARISON
# ============================================================================

def build_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Build multiple models and compare performance"""
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=20, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                   scoring='r2')
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred_test
        }
        
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test RMSE: ${test_rmse:.2f}")
        print(f"Test MAE: ${test_mae:.2f}")
        print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return results, scaler

# ============================================================================
# PART 6: MODEL VISUALIZATION & COMPARISON
# ============================================================================

def visualize_model_results(results):
    """Create comparison visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    model_names = list(results.keys())
    
    # R² Scores
    train_r2s = [results[m]['train_r2'] for m in model_names]
    test_r2s = [results[m]['test_r2'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, train_r2s, width, label='Train R²', color='skyblue', edgecolor='black')
    axes[0, 0].bar(x + width/2, test_r2s, width, label='Test R²', color='coral', edgecolor='black')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Model Comparison: R² Scores', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # RMSE Scores
    rmses = [results[m]['test_rmse'] for m in model_names]
    axes[0, 1].bar(model_names, rmses, color='lightgreen', edgecolor='black')
    axes[0, 1].set_ylabel('RMSE ($)')
    axes[0, 1].set_title('Model Comparison: Test RMSE', fontsize=14, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # MAE Scores
    maes = [results[m]['mae'] for m in model_names]
    axes[1, 0].bar(model_names, maes, color='lightyellow', edgecolor='black')
    axes[1, 0].set_ylabel('MAE ($)')
    axes[1, 0].set_title('Model Comparison: Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Cross-validation scores
    cv_means = [results[m]['cv_mean'] for m in model_names]
    cv_stds = [results[m]['cv_std'] for m in model_names]
    axes[1, 1].errorbar(model_names, cv_means, yerr=cv_stds, fmt='o-', 
                        markersize=10, capsize=5, capthick=2, linewidth=2, color='purple')
    axes[1, 1].set_ylabel('CV R² Score')
    axes[1, 1].set_title('Model Comparison: Cross-Validation Scores', fontsize=14, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("Model comparison visualization saved as 'model_comparison.png'")
    plt.show()

# ============================================================================
# PART 7: FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def analyze_feature_importance(results, feature_names):
    """Analyze and visualize feature importance"""
    
    # Get best model (usually Random Forest or XGBoost)
    best_model_name = max(results, key=lambda x: results[x]['test_r2'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Test R²: {results[best_model_name]['test_r2']:.4f}")
    
    # Extract feature importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        
        # Sort features
        indices = np.argsort(importances)[-20:]  # Top 20 features
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(len(indices)), importances[indices], color='teal', edgecolor='black')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top 20 Feature Importance ({best_model_name})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("Feature importance visualization saved as 'feature_importance.png'")
        plt.show()
        
        return importances
    
    return None

# ============================================================================
# PART 8: PREDICTION ANALYSIS
# ============================================================================

def analyze_predictions(results, y_test):
    """Analyze model predictions vs actual values"""
    
    best_model_name = max(results, key=lambda x: results[x]['test_r2'])
    y_pred = results[best_model_name]['predictions']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.5, s=20, color='blue')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Price ($)')
    axes[0].set_ylabel('Predicted Price ($)')
    axes[0].set_title(f'Actual vs Predicted Prices ({best_model_name})', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=20, color='green')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Price ($)')
    axes[1].set_ylabel('Residuals ($)')
    axes[1].set_title(f'Residual Plot ({best_model_name})', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions_analysis.png', dpi=300, bbox_inches='tight')
    print("Predictions analysis visualization saved as 'predictions_analysis.png'")
    plt.show()

# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Execute complete analysis pipeline"""
    
    print("\n" + "="*60)
    print("AIRBNB PRICING PREDICTION - COMPLETE PIPELINE")
    print("="*60 + "\n")
    
    # Step 1: Load Data
    print("Step 1: Loading Data...")
    # df = load_data('listings.csv')  # Replace with your dataset
    # For demonstration:
    print("Please load your Airbnb listings CSV file")
    print("Expected columns: price, room_type, accommodates, bedrooms, bathrooms, etc.\n")
    
    # Step 2: Clean Data
    # df_clean = clean_data(df)
    
    # Step 3: EDA
    # exploratory_analysis(df_clean)
    
    # Step 4: Feature Engineering
    # df_features, le_dict = feature_engineering(df_clean)
    
    # Step 5: Prepare for modeling
    # X = df_features.drop(['price', 'price_category'], axis=1)
    # y = df_features['price']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 6: Build Models
    # results, scaler = build_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Step 7: Visualize Results
    # visualize_model_results(results)
    
    # Step 8: Feature Importance
    # analyze_feature_importance(results, X_train.columns.tolist())
    
    # Step 9: Prediction Analysis
    # analyze_predictions(results, y_test)
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION GUIDE")
    print("="*60)
    print("""
To use this pipeline with your data:

1. Download Airbnb data from Inside Airbnb (https://insideairbnb.com/get-the-data/)
   or from Kaggle (https://www.kaggle.com/datasets/)

2. Update the load_data() function with your CSV file path

3. Uncomment all the main() function calls (lines starting with # df, results, etc.)

4. Run: python airbnb_pipeline.py

5. Check generated visualizations:
   - eda_analysis.png
   - model_comparison.png
   - feature_importance.png
   - predictions_analysis.png

Key parameters to adjust:
- test_size in train_test_split (0.2 = 20% test data)
- max_depth in tree-based models (higher = more complex)
- learning_rate in boosting models (lower = slower but better generalization)
- cv=5 for 5-fold cross-validation
    """)

if __name__ == "__main__":
    main()
