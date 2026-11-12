# FPL Model Training Guide

## 🎯 What You'll Learn

This comprehensive guide teaches you how to build a production-quality FPL points prediction model from scratch. You'll learn:

1. **Feature Selection** - How to choose the best features for your model
2. **Model Comparison** - How to evaluate different algorithms
3. **Hyperparameter Tuning** - How to fine-tune models for best performance
4. **Time-Series Validation** - How to prevent data leakage in temporal data
5. **Confidence Scoring** - How to quantify prediction uncertainty
6. **Multi-Horizon Forecasting** - How to predict 3 gameweeks ahead

## 📚 Notebook Structure

### `fpl_model_training_complete.ipynb`

A complete, educational notebook with 14 sections:

1. **Setup and Data Loading** - Import libraries and load your dataset
2. **Data Structure Analysis** - Understand your data
3. **Anti-Leakage Audit** - Prevent future data from contaminating training
4. **Multi-Horizon Target Creation** - Build GW+1, GW+2, GW+3 targets
5. **Feature Selection** - 3-stage pipeline to select best features
6. **Time-Series Cross-Validation** - Proper temporal validation
7. **Position-Specific Training** - Separate models per position
8. **Model Training** - Train all models for GW+1
9. **Performance Visualization** - Understand model behavior
10. **Hyperparameter Tuning** - Optimize model settings
11. **Feature Importance** - Understand what drives predictions
12. **Multi-Horizon Training** - Complete GW+2 and GW+3
13. **Confidence Scoring** - Quantify prediction uncertainty
14. **Summary and Next Steps** - Save models and review learnings

## 🚀 Quick Start

### 1. Activate Your Environment

```bash
source /home/rthomas/PersonalProjects/fpl-predictor/fpl_venv/bin/activate
```

### 2. Open the Notebook

Navigate to:
```
fantasy_football_scout/notebooks/fpl_model_training_complete.ipynb
```

### 3. Run All Cells

Execute cells sequentially from top to bottom. The notebook is designed to be self-contained and educational.

**Expected Runtime:** 15-30 minutes depending on your machine

## 📊 What Models Are Trained?

### Position-Specific Models
- **Goalkeeper** models (GW+1, GW+2, GW+3)
- **Defender** models (GW+1, GW+2, GW+3)
- **Midfielder** models (GW+1, GW+2, GW+3)
- **Forward** models (GW+1, GW+2, GW+3)

**Total: 12 models** (4 positions × 3 horizons)

### Algorithms Compared
1. **Ridge Regression** (Linear with L2 regularization)
2. **Lasso Regression** (Linear with L1 regularization)
3. **ElasticNet** (Linear with L1+L2 regularization)
4. **Random Forest** (Ensemble of decision trees)
5. **Gradient Boosting** (Sequential tree boosting)

## 🎓 Key Concepts Explained

### 1. Feature Selection

**Why?** Using all features often leads to overfitting. Select the most relevant ones.

**Methods:**
- **Variance Threshold**: Remove constant/near-constant features
- **Correlation Filter**: Remove redundant highly-correlated features
- **Mutual Information**: Measure statistical dependency with target

### 2. Time-Series Cross-Validation

**Why?** Random splits leak future data into training. Always train on PAST, test on FUTURE.

**How?**
```
Fold 1: train [----]     test [--]
Fold 2: train [------]   test [--]
Fold 3: train [--------] test [--]
```

### 3. Anti-Leakage Audit

**What?** Ensuring no information from the target match contaminates features.

**Examples:**
- ❌ BAD: Using `mins` (minutes played) - we don't know this pre-deadline
- ✅ GOOD: Using `form_mins` (historical average minutes)

### 4. Hyperparameter Tuning

**What?** Hyperparameters are model settings (not learned from data).

**Examples:**
- Random Forest: `n_estimators`, `max_depth`, `min_samples_split`
- Gradient Boosting: `learning_rate`, `n_estimators`, `max_depth`
- Ridge: `alpha` (regularization strength)

**Methods:**
- **Grid Search**: Try all combinations (exhaustive)
- **Random Search**: Try random combinations (faster)

### 5. Confidence Scoring

**What?** Not all predictions are equally reliable. Confidence scores (0-1) indicate reliability.

**How?** Based on historical prediction error patterns:
- High confidence = historically accurate predictions
- Low confidence = historically uncertain predictions

## 📈 Understanding Metrics

### RMSE (Root Mean Squared Error)
- Measures average prediction error
- Same units as target (FPL points)
- **Lower is better**
- Penalizes large errors more than small ones

### MAE (Mean Absolute Error)
- Average absolute prediction error
- More robust to outliers than RMSE
- **Lower is better**

### R² (R-squared)
- Proportion of variance explained by model
- Range: -∞ to 1
- **Higher is better**
- 0 = no better than predicting the mean
- 1 = perfect predictions

## 🔍 Interpreting Results

### Good Performance Indicators
- ✅ Test RMSE < 3.0 points for GW+1
- ✅ R² > 0.3 (explains >30% of variance)
- ✅ High-confidence predictions have lower errors
- ✅ Feature importance makes football sense

### Warning Signs
- ⚠️ Test performance much worse than CV performance (overfitting)
- ⚠️ Very high R² on training (>0.9) but low on test (overfitting)
- ⚠️ Confidence not correlated with error (poor calibration)
- ⚠️ Top features don't make sense (data issues)

## 🛠️ Customization Options

### Adjust Feature Selection
```python
# In train_position_horizon function
selected_features, selection_info = select_features(
    X, y, 
    max_features=25,  # Try 15, 20, 30
    variance_threshold=0.01,  # Try 0.001, 0.1
    correlation_threshold=0.95  # Try 0.85, 0.90
)
```

### Change Cross-Validation Splits
```python
# In time_series_evaluate function
tscv = TimeSeriesSplit(n_splits=5)  # Try 3, 7, 10
```

### Modify Models
```python
models = {
    'Ridge': Ridge(alpha=1.0),  # Try alpha=0.1, 10.0
    'RandomForest': RandomForestRegressor(
        n_estimators=100,  # Try 50, 200, 500
        max_depth=10,  # Try 5, 15, 20, None
        random_state=42
    ),
}
```

## 💾 Saved Outputs

After running the notebook, you'll have:

### Models Directory (`../models/`)
- `midfielder_gw1_randomforest.joblib` - Trained models
- `midfielder_gw1_scaler.joblib` - Feature scalers (for linear models)
- `midfielder_gw1_features.txt` - Selected feature lists

### Data Directory (`../data/`)
- `model_summary.csv` - Performance summary for all models

## 🔄 Next Steps After Training

### 1. Production Inference Script
Create a script to make predictions on new data:
```python
import joblib
import pandas as pd

# Load model and features
model = joblib.load('models/midfielder_gw1_randomforest.joblib')
features = open('models/midfielder_gw1_features.txt').read().split('\n')

# Load new data
new_data = pd.read_csv('current_gameweek_data.csv')

# Prepare features
X = new_data[features].fillna(new_data[features].median())

# Predict
predictions = model.predict(X)
```

### 2. Advanced Feature Engineering
- Fixture difficulty ratings
- Recent form trends (last 3, 5, 10 games)
- Head-to-head history vs opponent
- Home/away split performance
- Injury/suspension flags

### 3. Try Advanced Models
- **XGBoost**: Often best for tabular data
- **LightGBM**: Fast and memory efficient
- **CatBoost**: Great for categorical features
- **Neural Networks**: Can capture complex patterns

### 4. Ensemble Methods
Combine multiple models for better predictions:
```python
# Simple average ensemble
pred_rf = model_rf.predict(X)
pred_gb = model_gb.predict(X)
pred_ensemble = (pred_rf + pred_gb) / 2
```

### 5. Online Learning
Update models weekly as new data arrives:
```python
# Incremental update (for some models)
model.partial_fit(X_new, y_new)
```

## ❓ Troubleshooting

### Issue: Out of Memory
**Solution:** Reduce `max_features` or use fewer samples for hyperparameter tuning

### Issue: Training Too Slow
**Solution:** 
- Reduce `n_iter` in RandomizedSearchCV
- Use fewer cross-validation splits
- Use `n_jobs=-1` for parallel processing

### Issue: Poor Performance
**Solution:**
- Check for data leakage (review anti-leakage audit)
- Ensure proper time-series ordering
- Try different feature sets
- Gather more training data

### Issue: Models Overfit
**Solution:**
- Increase regularization (higher alpha for linear models)
- Reduce model complexity (lower max_depth for trees)
- Use more cross-validation folds
- Select fewer features

## 📚 Additional Resources

### Scikit-learn Documentation
- [User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Model Selection](https://scikit-learn.org/stable/model_selection.html)
- [Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)

### Time Series Forecasting
- [Forecasting: Principles and Practice](https://otexts.com/fpp2/)
- [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

### Feature Engineering
- [Feature Engineering and Selection](https://bookdown.org/max/FES/)
- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)

### Advanced Topics
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Hyperparameter Optimization](https://scikit-optimize.github.io/)

## 🚀 Ready to Start?

1. Open `fpl_model_training_complete.ipynb`
2. Execute cells from top to bottom
3. Read the explanations and visualizations
4. Experiment with parameters
5. Build your intuition!

**Remember:** Machine learning is iterative. Don't expect perfect results on the first try. Experiment, learn, and improve! 🎯
