# High-Confidence Stocks AI Training Guide

## 🎯 Training Status

**Training Script:** `train_8499_high_confidence.py`
**Status:** Running in background (Task ID: b4b1149)

---

## 📊 Stocks Being Trained (In Order)

### 1. 8499.TW - 鼎炫-KY (Electronic Test Equipment)
- **Test Accuracy**: 74.4%
- **Confidence**: ⭐⭐⭐⭐⭐ 最高
- **Overfitting Gap**: 25.17% (Low - Good!)
- **Why High Confidence**: Best performer, lowest overfitting

### 2. 2408.TW - 南亚科 (DRAM Manufacturer)
- **Test Accuracy**: 65.1%
- **Confidence**: ⭐⭐⭐⭐ 高
- **Overfitting Gap**: 34.47% (Medium-Low)
- **Why High Confidence**: Strong accuracy, semiconductor cycle predictability

### 3. 2368.TW - 金像电 (PCB Manufacturer)
- **Test Accuracy**: 55.8%
- **Confidence**: ⭐⭐⭐ 中高
- **Overfitting Gap**: 43.77% (Medium)
- **Why Moderate Confidence**: Stable business, moderate accuracy

### 4. 2383.TW - 台光电 (CCL - Copper Clad Laminate)
- **Test Accuracy**: 55.8%
- **Confidence**: ⭐⭐⭐ 中高
- **Overfitting Gap**: 43.77% (Medium)
- **Why Moderate Confidence**: Material supplier, steady demand

### 5. 4722.TW - 国精化 (Semiconductor Chemicals)
- **Test Accuracy**: 55.8%
- **Confidence**: ⭐⭐⭐ 中高
- **Overfitting Gap**: 44.19% (Medium)
- **Why Moderate Confidence**: Chemical materials, consistent business

---

## 🚀 Training Configuration

### Data Settings:
- **Historical Period**: 2015-01-01 to 2025-01-12 (10 years)
- **Train/Test Split**: 80% / 20%
- **Total Timesteps**: 150,000 per stock
- **Estimated Training Time**: ~30-45 minutes per stock

### Model Architecture:
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Action Space**: Continuous [-1.0, 1.0]
  - -1.0 = Sell all
  - 0.0 = Hold
  - +1.0 = Buy all
- **Observation Space**: 15 features
- **Policy**: MlpPolicy (Multi-Layer Perceptron)

### Reward Function:
```python
reward = profit_reward + trade_incentive + cash_penalty

Where:
- profit_reward = (current_value - initial_value) / initial_value
- trade_incentive = 0.01 (if action > 0.1, encourages trading)
- cash_penalty = -0.005 (if cash > 90%, discourages hoarding cash)
```

### Hyperparameters:
- Learning Rate: 0.0003
- N Steps: 2,048
- Batch Size: 64
- N Epochs: 10
- Gamma: 0.99
- Entropy Coefficient: 0.01

---

## 📁 Output Files (Per Stock)

For each stock (e.g., 8499.TW), the following files will be generated:

### 1. Model File
```
ppo_8499_tw_improved.zip
```
- Trained PPO model ready for trading
- Can be loaded with `model = PPO.load("ppo_8499_tw_improved")`

### 2. Feature Importance JSON
```
8499_TW_feature_importance.json
```
- Contains ML analysis of which technical indicators matter most
- Format:
```json
{
  "ticker": "8499.TW",
  "analysis_date": "2026-01-12",
  "model_accuracy": 0.7442,
  "feature_importance": {
    "ATR": 0.2082,
    "MA_200": 0.1415,
    ...
  }
}
```

### 3. Feature Importance Chart
```
8499_TW_feature_importance.png
```
- Visual bar chart showing which indicators are most predictive
- Helps understand what drives the stock's movement

---

## 📈 Technical Indicators Used (17 Total)

### Trend Indicators:
1. **SMA 10, 30, 50** - Short, medium, long-term trends
2. **MA 20, 50, 200** - Moving averages
3. **MA50_slope** - Trend direction and strength

### Momentum Indicators:
4. **RSI** - Relative Strength Index (overbought/oversold)
5. **MACD** - Moving Average Convergence Divergence
6. **MACD_signal** - Signal line
7. **MACD_hist** - Histogram

### Volatility Indicators:
8. **ATR** - Average True Range
9. **Volatility** - Price volatility
10. **BB Position** - Bollinger Band position

### Volume Indicators:
11. **OBV** - On-Balance Volume
12. **OBV_MA** - OBV moving average

### Oscillators:
13. **K** - Stochastic K%
14. **D** - Stochastic D%

### Price Change:
15. **price_change_5d** - 5-day return
16. **price_change_20d** - 20-day return

---

## 🔍 Monitoring Training Progress

### Check Current Output:
```bash
# Read the training log
.venv/Scripts/python.exe -c "import sys; print(open(r'C:\Users\Silvi\AppData\Local\Temp\claude\c--Users-Silvi-Projects-trading-bot\tasks\b4b1149.output', encoding='utf-8').read())"
```

### Or use tail to see recent output:
```bash
tail -50 C:\Users\Silvi\AppData\Local\Temp\claude\c--Users-Silvi-Projects-trading-bot\tasks\b4b1149.output
```

### Expected Output Stages:
1. ✅ Downloading stock data
2. ✅ Adding technical indicators
3. ✅ Feature importance analysis
4. ✅ Model training (progress updates every few thousand steps)
5. ✅ Model saved

---

## 📊 After Training Completes

### For Each Stock You'll Have:

**8499.TW (鼎炫-KY):**
- `ppo_8499_tw_improved.zip` - Trading model
- `8499_TW_feature_importance.json` - Feature analysis
- `8499_TW_feature_importance.png` - Visual chart

**2408.TW (南亚科):**
- `ppo_2408_tw_improved.zip`
- `2408_TW_feature_importance.json`
- `2408_TW_feature_importance.png`

*(And similarly for 2368, 2383, 4722)*

---

## 🎯 Next Steps After Training

### 1. Backtesting
Test the trained models on historical test data:
```python
from stable_baselines3 import PPO

# Load model
model = PPO.load("ppo_8499_tw_improved")

# Test on test_df
env = ImprovedTradingEnv(test_df)
obs, _ = env.reset()

for _ in range(len(test_df)):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    if done:
        break

print(f"Final Profit: ${env.total_profit:.2f}")
```

### 2. Paper Trading
Run live simulation without real money for 1-2 weeks

### 3. Real Trading (If Results Good)
- Start with small position sizes (1-2% of portfolio)
- Monitor performance daily
- Set stop-losses

---

## ⚠️ Important Notes

### Training Time Estimates:
- **Per Stock**: ~30-45 minutes
- **Total (5 stocks)**: ~2.5-4 hours

### Resource Usage:
- CPU-intensive (uses all cores)
- Memory: ~2-4 GB per stock
- Disk: ~50-100 MB per model

### When Training Completes:
You'll see output like:
```
✅ 8499.TW (鼎炫-KY) 训练完成!
   模型文件: ppo_8499_tw_improved.zip
   特征重要性: 8499_TW_feature_importance.json

🎉 所有高信心股票训练完成!
```

---

## 🔧 Troubleshooting

### If Training Fails:
1. Check the output file for errors
2. Common issues:
   - Data download failed → Check internet connection
   - Out of memory → Close other applications
   - Import errors → Check dependencies installed

### Re-run Training:
```bash
python train_8499_high_confidence.py
```

---

## 📚 Reference Documents

Related files created:
1. `MODEL_IMPROVEMENT_SUMMARY.md` - V1.0 improvements
2. `ACCURACY_REPORT.md` - Full accuracy analysis
3. `V2_COMPARISON.md` - V1 vs V2 comparison
4. `8499_MODEL_ANALYSIS.md` - Detailed 8499.TW analysis

---

## ✅ Success Criteria

**Training is successful if:**
1. ✅ All 5 models saved (.zip files created)
2. ✅ All feature importance files generated (.json + .png)
3. ✅ No error messages in output
4. ✅ Each model shows training progress to 150,000 steps

**Model is ready for production if:**
1. ✅ Backtest shows positive returns on test set
2. ✅ Paper trading successful for 2+ weeks
3. ✅ Maximum drawdown acceptable (<20%)
4. ✅ Win rate + risk/reward ratio profitable

---

## 🎓 Model Quality Expectations

### Based on ML Analysis:

**8499.TW:**
- Expected: Best performer
- Real-world accuracy: ~50-52% (realistic)
- Edge: ATR signals + mean reversion

**2408.TW:**
- Expected: Strong performer
- Real-world accuracy: ~50-51%
- Edge: Semiconductor cycle timing

**Others (2368, 2383, 4722):**
- Expected: Moderate performance
- Real-world accuracy: ~50-51%
- Edge: Steady business patterns

**Remember:** 52% accuracy with 2:1 R/R = Very profitable!

---

## 📞 Support

If training completes successfully:
- Check output files in project directory
- Load models and test
- Review feature importance to understand what matters

If issues occur:
- Check task output file
- Verify dependencies installed
- Ensure sufficient disk space/memory

**Training started at**: 2026-01-12
**Expected completion**: ~2.5-4 hours from start
