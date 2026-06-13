# 🎯 AI Trading Signal Accuracy Improvement Plan

## Current Accuracy Status

**Top Performers:**
- 6220.TWO (岳豐): 75.60% ⭐⭐ EXCELLENT
- 6877.TWO (鏵友益): 70.75% ⭐⭐ EXCELLENT

**Average Performers (50-65%):**
- 3576.TW, 3615.TWO: ~68%
- 8927.TWO: ~67%
- 6230.TW: ~65%
- 4564.TW, 4989.TW: ~64-65%
- **Current 6531.TW: 54.7%**

**Below Average (<50%):**
- 6526.TW (達發): 49.28% ⚠️
- 6442.TW, 4768.TWO: ~50%
- 4577.TWO: 51.42%

---

## 🔧 Improvement Strategies

### 1. **Feature Engineering Enhancements** (High Impact)
   
**Current Features (19):**
- SMA (10, 30, 50), EMA (12, 26)
- RSI, MACD, Bollinger Bands
- Price changes, Volume ratio, High-Low range

**Recommendations to Add:**
```
Technical Indicators:
  - ADX (Average Directional Index) - Trend strength
  - Stochastic RSI - Momentum confirmation
  - OBV (On-Balance Volume) - Volume trend
  - Williams %R - Overbought/oversold levels
  - CCI (Commodity Channel Index) - Mean reversion
  - ATR (Average True Range) - Volatility measure

Market Microstructure:
  - Order flow imbalance - Buy/sell pressure
  - Bid-ask spread - Liquidity measure
  - Volume momentum - Acceleration detection

Sentiment & Fundamentals:
  - News sentiment score - Market psychology
  - PE ratio momentum - Valuation changes
  - Earnings surprises - Forward expectations
```

**Expected Impact:** +5-10% accuracy improvement

---

### 2. **Model Retraining & Optimization** (High Impact)

**Current Issue:** Models last trained 2026-03-02 (27 days old)
- Market conditions change constantly
- Models become stale without recent data

**Recommended Strategy:**
```
Phase 1: Immediate Actions
  ✓ Retrain models weekly with last 180 days of data
  ✓ Use rolling window validation (k-fold: 5)
  ✓ Implement early stopping to prevent overfitting

Phase 2: Advanced Techniques
  ✓ Hyperparameter optimization (GridSearch/RandomSearch)
  ✓ Test different tree depths: [4, 6, 8, 10]
  ✓ Adjust learning rate: [0.01, 0.05, 0.1]
  ✓ Experiment with subsample: [0.6, 0.8, 1.0]
```

**Code Example:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

grid_search = GridSearchCV(
    xgb.XGBClassifier(),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
```

**Expected Impact:** +3-8% accuracy improvement

---

### 3. **Ensemble Methods** (Medium Impact)

**Current Method:** Single XGBoost model
**Improved Method:** Model voting ensemble

```
Ensemble Strategy:
  Model 1: XGBoost (Classification)
  Model 2: LightGBM (Speed + accuracy)
  Model 3: CatBoost (Categorical handling)
  Model 4: Random Forest (Robustness)
  
Voting Method: Hard voting (majority wins)
  - BUY if 3+ models agree
  - SELL if 3+ models agree  
  - HOLD if split decision
```

**Code Example:**
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model),
        ('rf', rf_model)
    ],
    voting='hard'
)
```

**Expected Impact:** +2-5% accuracy improvement

---

### 4. **Market Regime Detection** (High Impact)

**Problem:** Same model works differently in bull/bear markets
**Solution:** Use separate models for different market regimes

```
Market States:
  1. Strong Uptrend (SMA10 > SMA30 > SMA50)
     → Use aggressive BUY threshold (35%)
     → Higher confidence in positive signals
  
  2. Downtrend (SMA10 < SMA30 < SMA50)
     → Use conservative BUY threshold (70%)
     → Focus on SELL signals
  
  3. Sideways/Consolidation (SMA convergence)
     → HOLD preference
     → Wider bands before trading
```

**Expected Impact:** +4-7% accuracy improvement

---

### 5. **Risk-Adjusted Signal Filtering** (High Impact)

**Current Issue:** Signals are issued without considering volatility/liquidity

**Recommended Filters:**

```
Filter 1: Volatility Check
  - Calculate 20-day ATR
  - Only trade if ATR < 5% (normal volatility)
  - Skip if ATR > 10% (high volatility = high risk)

Filter 2: Liquidity Check
  - Minimum volume: 1,000,000 shares/day
  - Bid-ask spread < 1%
  - Skip low-liquidity stocks

Filter 3: Technical Confirmation
  - Require 2+ indicators in agreement
  - RSI + MACD + Price Action alignment
  - Reduce false signals by 20-30%

Filter 4: Time-based Filters
  - Avoid trading in last 30 min (closing volatility)
  - Avoid earnings date ±3 days
  - Prefer continuation patterns over reversals
```

**Expected Impact:** +3-5% accuracy (via signal quality)

---

### 6. **Machine Learning Optimization** (Medium Impact)

**Cost Function Weighting:**
```python
# Current: Equal weight to BUY/SELL errors
# Improved: Penalize costly errors more

class_weight = {
    0: 1.0,    # DOWN (SELL)
    1: 1.5     # UP (BUY) - weight more heavily
}

model = xgb.XGBClassifier(
    scale_pos_weight=1.5,  # or use class_weight
    eval_metric='logloss'
)
```

**Data Imbalance Handling:**
```python
from imblearn.over_sampling import SMOTE

# Handle imbalanced classes
smote = SMOTE(sampling_strategy=0.8)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

**Expected Impact:** +2-4% accuracy improvement

---

### 7. **Probabilistic Thresholding** (High Impact)

**Current Method:** Binary prediction (>50% = BUY)
**Improved Method:** Confidence-weighted signals

```
Signal Strength Strategy:
  Prediction Confidence > 75%: Strong signal (confidence=0.8-1.0)
  Prediction Confidence 65-75%: Medium signal (confidence=0.5-0.7)
  Prediction Confidence 55-65%: Weak signal (confidence=0.2-0.4)
  Prediction Confidence < 55%: HOLD (no action)

Action taken based on confidence:
  Strong BUY (75%+):     Position size 100%
  Medium BUY (65-75%):   Position size 50%
  Weak BUY (55-65%):     Position size 25%
  Weak SELL (55-65%):    Reduce by 25%
  Medium SELL (65-75%):  Reduce by 50%
  Strong SELL (75%+):    Exit full position
```

**Expected Impact:** +2-3% risk-adjusted returns

---

### 8. **Data Quality & Preprocessing** (Medium Impact)

**Current Issues to Address:**
```
✓ Missing data handling (use forward fill, not drop)
✓ Outlier detection (Z-score > 3σ)
✓ Normalization/Standardization (StandardScaler)
✓ Feature scaling consistency
✓ Data leakage prevention
```

**Improvements:**
```python
# Robust outlier handling
from scipy import stats

for col in numeric_cols:
    z_scores = np.abs(stats.zscore(df[col].dropna()))
    df_clean = df[(z_scores < 3).all(axis=1)]

# Proper scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

**Expected Impact:** +1-2% accuracy improvement

---

## 📊 Implementation Priority & Timeline

**Week 1-2 (Immediate):**
- ✅ Add 5-6 new technical indicators (ADX, Stochastic, OBV)
- ✅ Implement weekly retraining schedule
- ✅ Add risk filters (volatility, liquidity, technical confirmation)

**Week 3-4 (High Priority):**
- ✅ Implement ensemble with LightGBM + CatBoost
- ✅ Market regime detection
- ✅ Probabilistic thresholding

**Week 5-6 (Enhancement):**
- ✅ Hyperparameter optimization
- ✅ Data imbalance handling (SMOTE)
- ✅ Advanced feature engineering

**Ongoing:**
- ✅ Weekly model retraining
- ✅ Backtesting on out-of-sample data
- ✅ Live performance monitoring

---

## 🎯 Expected Results

**Conservative Estimate:**
- Current Average Accuracy: ~59%
- Projected After All Improvements: **70-75%**
- **Improvement: +11-16%**

**Optimistic Scenario:**
- Could reach: **75-80%** with perfect implementation

**Timeline:** 4-8 weeks to full implementation

---

## ⚠️ Important Notes

1. **No Model is Perfect:** 80% accuracy is excellent in trading
2. **Out-of-Sample Testing:** Always validate on new data
3. **Market Regime Changes:** Retraining is ongoing, not one-time
4. **Risk Management:** Accuracy ≠ Profitability (position sizing matters)
5. **Overfitting Risk:** Use proper cross-validation to prevent

---

## 📈 Quick Wins (Implement First)

1. Add volatility filter (3-5% improvement)
2. Ensemble voting (2-5% improvement)
3. Weekly retraining (5-10% improvement)
4. Probabilistic thresholding (2-3% improvement)

**Combined Quick Wins Expected Impact: +12-23%**

---

## 🔗 Next Steps

1. Choose which improvements to implement first
2. Test on historical data (backtesting)
3. Paper trade for 2-4 weeks
4. Monitor performance vs. current models
5. Adjust parameters based on live results
