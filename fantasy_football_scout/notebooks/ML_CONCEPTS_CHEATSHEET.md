# ML Concepts Quick Reference

## 🎯 Feature Selection

### Why Feature Selection?
- ✅ Reduces overfitting (simpler models generalize better)
- ✅ Faster training and prediction
- ✅ Easier to interpret
- ✅ Removes noise and redundancy

### Methods

#### 1. Variance Threshold
Removes features with low variance (nearly constant values).
```python
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)
```

#### 2. Correlation Filter
Removes highly correlated features (redundant information).
```python
# Find pairs with correlation > 0.95
corr_matrix = X.corr().abs()
to_drop = []  # features to remove
```

#### 3. Mutual Information
Measures statistical dependency between features and target.
```python
from sklearn.feature_selection import mutual_info_regression
mi_scores = mutual_info_regression(X, y)
# Higher score = more relevant feature
```

#### 4. Model-Based (Feature Importance)
Uses model to rank features by importance.
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X, y)
importances = model.feature_importances_
```

---

## ⏱️ Time-Series Cross-Validation

### Why NOT Random Split?
Random splits cause **temporal leakage**:
- Future data in training set
- Past data in test set
- Unrealistic evaluation (you can't use future to predict past!)

### TimeSeriesSplit
Always trains on PAST, tests on FUTURE.

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # Train and evaluate
```

**Visualization:**
```
Fold 1: train [XXXX----] test [XX]
Fold 2: train [XXXXXX--] test [XX]
Fold 3: train [XXXXXXXX] test [XX]
```

---

## 🔒 Data Leakage Prevention

### What is Leakage?
Information from the **target** (what we're predicting) leaking into **features**.

### Types of Leakage

#### 1. Target Leakage
Using the target variable itself as a feature.
```python
# ❌ BAD: Using points from the match we're predicting
X['points_this_match'] = y  

# ✅ GOOD: Using historical points
X['points_last_5_matches'] = historical_points
```

#### 2. Temporal Leakage
Using future information in training.
```python
# ❌ BAD: Computing feature using ALL data (including future)
X['season_avg'] = df.groupby('player')['points'].transform('mean')

# ✅ GOOD: Computing feature using only PAST data
X['rolling_avg'] = df.groupby('player')['points'].rolling(5).mean()
```

#### 3. Train-Test Contamination
Fitting transformations on ALL data before splitting.
```python
# ❌ BAD: Scaling before split
X_scaled = scaler.fit_transform(X)  # Uses info from test set!
X_train, X_test = train_test_split(X_scaled)

# ✅ GOOD: Scaling after split
X_train, X_test = train_test_split(X)
scaler.fit(X_train)  # Learn scaling ONLY from training
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### FPL-Specific Leakage Examples

| Feature | Available Pre-Deadline? | Verdict |
|---------|------------------------|---------|
| `form_points` (historical) | ✅ Yes | Safe |
| `class_xg` (historical) | ✅ Yes | Safe |
| `team`, `opponent` | ✅ Yes | Safe |
| `mins` (from target match) | ❌ No | LEAKAGE! |
| `starts` (from target match) | ❌ No | LEAKAGE! |
| `goals` (from target match) | ❌ No | LEAKAGE! |

---

## 🎛️ Hyperparameter Tuning

### What Are Hyperparameters?
Settings that control HOW a model learns (not learned from data).

### Common Hyperparameters

#### Random Forest
```python
RandomForestRegressor(
    n_estimators=100,     # Number of trees (more = better, slower)
    max_depth=10,         # Tree depth (higher = more complex)
    min_samples_split=2,  # Min samples to split node
    min_samples_leaf=1,   # Min samples in leaf
    max_features='sqrt'   # Features per split
)
```

#### Gradient Boosting
```python
GradientBoostingRegressor(
    n_estimators=100,     # Number of boosting stages
    learning_rate=0.1,    # Shrinkage (lower = more conservative)
    max_depth=5,          # Tree depth
    min_samples_split=2,
    subsample=0.8         # Fraction of samples per tree
)
```

#### Ridge/Lasso
```python
Ridge(alpha=1.0)  # Regularization strength (higher = simpler)
Lasso(alpha=0.1)
```

### Tuning Methods

#### Grid Search (Exhaustive)
Tries ALL combinations.
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}
# 3 × 3 = 9 combinations

grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
```

#### Random Search (Sample)
Tries RANDOM combinations.
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [50, 100, 150, 200, 300],
    'max_depth': [3, 5, 7, 10, 15, 20, None]
}

search = RandomizedSearchCV(
    model, param_dist, 
    n_iter=10,  # Try 10 random combinations
    cv=5
)
search.fit(X_train, y_train)
```

**Grid vs Random:**
- Grid: Exhaustive but slow
- Random: Faster but might miss optimal
- **Recommended:** Start with Random (quick), then Grid around best areas

---

## 📊 Evaluation Metrics

### RMSE (Root Mean Squared Error)
```python
import numpy as np
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```
- **Units:** Same as target (FPL points)
- **Interpretation:** Average prediction error
- **Range:** 0 to ∞ (0 = perfect)
- **Penalty:** Squares errors (large errors penalized more)

**Example:** RMSE = 2.5 means predictions are off by ~2.5 points on average.

### MAE (Mean Absolute Error)
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```
- **Units:** Same as target
- **Interpretation:** Average absolute error
- **Range:** 0 to ∞ (0 = perfect)
- **Penalty:** Linear (all errors weighted equally)

**Example:** MAE = 2.0 means absolute error is 2.0 points on average.

### R² (R-squared / Coefficient of Determination)
```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```
- **Units:** Dimensionless (proportion)
- **Interpretation:** % of variance explained by model
- **Range:** -∞ to 1
  - 1 = perfect predictions
  - 0 = no better than predicting mean
  - < 0 = worse than predicting mean

**Example:** R² = 0.35 means model explains 35% of variance in points.

### Which Metric to Use?

| Metric | Best For | Why |
|--------|----------|-----|
| RMSE | When large errors are very bad | Penalizes outliers heavily |
| MAE | When all errors matter equally | Interpretable, robust |
| R² | Understanding model quality | Easy to communicate (%) |

**Recommendation:** Report all three! They tell different stories.

---

## 🤖 Model Selection Guide

### Linear Models (Ridge, Lasso, ElasticNet)

**When to use:**
- ✅ Large datasets (fast training)
- ✅ Want interpretability
- ✅ Linear relationships
- ✅ Need baseline quickly

**Pros:**
- Fast training and prediction
- Low overfitting risk
- Interpretable coefficients

**Cons:**
- Can't capture non-linear patterns
- May underfit complex data

**Hyperparameters:**
- `alpha`: Regularization (0.01 to 100)

### Random Forest

**When to use:**
- ✅ Non-linear relationships
- ✅ Want feature importances
- ✅ Medium-sized datasets
- ✅ Robust to outliers

**Pros:**
- Handles non-linearity well
- Robust to outliers and scaling
- Built-in feature importance
- Parallel training

**Cons:**
- Can overfit (need tuning)
- Slower than linear models
- Memory intensive

**Hyperparameters:**
- `n_estimators`: 100-500
- `max_depth`: 5-20 (or None)
- `min_samples_split`: 2-10

### Gradient Boosting

**When to use:**
- ✅ Want best accuracy
- ✅ Willing to tune carefully
- ✅ Sequential patterns matter
- ✅ Don't need real-time predictions

**Pros:**
- Often best accuracy
- Handles complex patterns
- Feature importance

**Cons:**
- Easy to overfit (needs careful tuning)
- Slow training (sequential)
- Sensitive to hyperparameters

**Hyperparameters:**
- `n_estimators`: 100-300
- `learning_rate`: 0.01-0.2
- `max_depth`: 3-7
- `subsample`: 0.8-1.0

### Quick Decision Tree

```
Is accuracy most important? 
    ├─ Yes → GradientBoosting (tune carefully)
    └─ No
        ├─ Need fast predictions? → Ridge/Lasso
        ├─ Non-linear data? → RandomForest
        └─ Want baseline? → Ridge
```

---

## 🎲 Confidence Scoring

### Why Confidence?
Not all predictions are equally reliable. Confidence quantifies uncertainty.

### Methods

#### 1. Residual-Based (Simple)
Based on historical prediction errors.
```python
residuals = y_true - y_pred
abs_residuals = np.abs(residuals)
confidence = 1 - (abs_residuals / abs_residuals.max())
```

#### 2. Variance-Based (Ensemble)
Use prediction variance from ensemble models.
```python
# For RandomForest, get predictions from all trees
predictions_per_tree = [tree.predict(X) for tree in model.estimators_]
prediction_std = np.std(predictions_per_tree, axis=0)
confidence = 1 / (1 + prediction_std)
```

#### 3. Quantile Regression
Predict confidence intervals instead of point estimates.
```python
from sklearn.ensemble import GradientBoostingRegressor
# Train models for different quantiles
model_lower = GradientBoostingRegressor(loss='quantile', alpha=0.1)
model_upper = GradientBoostingRegressor(loss='quantile', alpha=0.9)
# Prediction interval: [lower, median, upper]
```

### Calibration Check
High confidence should correlate with low error.

```python
# Bin predictions by confidence
df['conf_bin'] = pd.qcut(confidence, q=5)
# Check: Higher confidence bins should have lower avg error
df.groupby('conf_bin')['error'].mean()
```

---

## 🔧 Common Issues & Solutions

### Issue: Overfitting
**Symptoms:** Great training performance, poor test performance

**Solutions:**
- ✅ Use cross-validation
- ✅ Increase regularization (alpha)
- ✅ Reduce model complexity (max_depth)
- ✅ Use fewer features
- ✅ Get more training data

### Issue: Underfitting
**Symptoms:** Poor training AND test performance

**Solutions:**
- ✅ Use more complex model
- ✅ Add more features
- ✅ Reduce regularization
- ✅ Engineer better features

### Issue: Slow Training
**Solutions:**
- ✅ Use `n_jobs=-1` for parallel processing
- ✅ Reduce dataset size (sample if huge)
- ✅ Use fewer features
- ✅ Use simpler models (Ridge vs RandomForest)

### Issue: Poor Predictions
**Checklist:**
- [ ] Check for data leakage
- [ ] Verify time-series ordering
- [ ] Inspect feature distributions
- [ ] Check for missing values
- [ ] Validate train/test split
- [ ] Review feature selection
- [ ] Try different models
- [ ] Tune hyperparameters

---

## 📖 Quick Command Reference

### Load Data
```python
df = pd.read_csv('data.csv')
```

### Feature Selection
```python
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression

# Remove low variance
selector = VarianceThreshold(threshold=0.01)
X_var = selector.fit_transform(X)

# Select top K by mutual information
selector = SelectKBest(mutual_info_regression, k=20)
X_selected = selector.fit_transform(X, y)
```

### Train-Test Split (Time-Series)
```python
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
```

### Scale Features
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Train Model
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### Evaluate
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

### Save/Load Model
```python
import joblib

# Save
joblib.dump(model, 'model.joblib')

# Load
model = joblib.load('model.joblib')
```

### Cross-Validation
```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]}
search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=5)
search.fit(X_train, y_train)
best_model = search.best_estimator_
```

---

## 💡 Pro Tips

1. **Start Simple**: Begin with Ridge regression baseline before complex models
2. **Visualize Everything**: Plot distributions, correlations, residuals
3. **Check Assumptions**: Ensure time-series ordering, no leakage
4. **Use Cross-Validation**: Never trust single train/test split
5. **Monitor Overfitting**: CV score vs test score should be similar
6. **Feature Engineering > Model Complexity**: Good features beat fancy models
7. **Iterate**: ML is iterative; don't expect perfection first try
8. **Document**: Save model parameters, feature lists, performance metrics
9. **Ensemble**: Combine multiple models for better predictions
10. **Domain Knowledge**: FPL expertise is as important as ML skills!

---

## 📚 Further Reading

- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Feature Engineering Book](https://www.feat.engineering/)
- [Forecasting: Principles and Practice](https://otexts.com/fpp2/)
- [Kaggle Learn](https://www.kaggle.com/learn)
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)
