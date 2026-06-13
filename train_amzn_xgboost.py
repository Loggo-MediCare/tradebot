"""
美股 AMZN (Amazon) XGBoost 交易模型訓練
使用 XGBoost 替代 PPO 以提高預測準確度
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import json
from datetime import datetime

print("=" * 80)
print("訓練 AMZN (Amazon) XGBoost 模型")
print("=" * 80)

# 1. 下載數據
print("\n下載股票數據...")
TICKER = 'AMZN'
df = yf.download(TICKER, start='2015-01-01', end='2025-12-31', progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

df = df.rename(columns={'Close': 'close', 'Volume': 'volume', 'Open': 'open',
                        'High': 'high', 'Low': 'low'}).reset_index()

print(f"成功下載 {len(df)} 天數據")
print(f"   價格範圍: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

# 2. 添加技術指標
print("\n計算技術指標...")
df['sma_10'] = df['close'].rolling(10).mean()
df['sma_30'] = df['close'].rolling(30).mean()
df['sma_50'] = df['close'].rolling(50).mean()
df['sma_200'] = df['close'].rolling(200).mean()

df['ema_12'] = df['close'].ewm(span=12).mean()
df['ema_26'] = df['close'].ewm(span=26).mean()

# RSI
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / (loss + 1e-10)
df['rsi'] = 100 - (100 / (1 + rs))

# MACD
df['macd'] = df['ema_12'] - df['ema_26']
df['macd_signal'] = df['macd'].ewm(span=9).mean()
df['macd_hist'] = df['macd'] - df['macd_signal']

# Bollinger Bands
df['bb_middle'] = df['close'].rolling(20).mean()
df['bb_std'] = df['close'].rolling(20).std()
df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100).fillna(50)

# KD
low_14 = df['low'].rolling(14).min()
high_14 = df['high'].rolling(14).max()
df['K'] = ((df['close'] - low_14) / (high_14 - low_14) * 100).fillna(50)
df['D'] = df['K'].rolling(3).mean()

# OBV
df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
df['obv_ma20'] = df['obv'].rolling(20).mean()

# 波動性
df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()

# ATR
high_low = df['high'] - df['low']
high_close = np.abs(df['high'] - df['close'].shift())
low_close = np.abs(df['low'] - df['close'].shift())
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['atr'] = true_range.rolling(14).mean()

# 價格變化率
df['price_change_5d'] = df['close'].pct_change(5) * 100
df['price_change_10d'] = df['close'].pct_change(10) * 100
df['price_change_20d'] = df['close'].pct_change(20) * 100

# MA50 斜率
df['ma50_slope'] = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100

# 3. 定義目標變量（未來5天漲跌）
df['future_return'] = df['close'].shift(-5) / df['close'] - 1
df['target'] = (df['future_return'] > 0.02).astype(int)  # 未來5天漲超過2%為買入信號

# 4. 準備特徵
feature_columns = [
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_position', 'K', 'D', 'obv', 'obv_ma20',
    'sma_10', 'sma_30', 'sma_50', 'sma_200',
    'volatility', 'atr',
    'price_change_5d', 'price_change_10d', 'price_change_20d',
    'ma50_slope'
]

# 移除缺失值
df_clean = df.dropna(subset=feature_columns + ['target'])
print(f"有效數據點: {len(df_clean)}")

X = df_clean[feature_columns]
y = df_clean['target']

# 5. 分割數據
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"\n數據分割:")
print(f"   訓練集: {len(X_train)} 筆")
print(f"   測試集: {len(X_test)} 筆")
print(f"   目標分布 - 買入: {y_train.sum()} ({y_train.mean()*100:.1f}%), 不買: {len(y_train)-y_train.sum()}")

# 6. 訓練 XGBoost 模型
print("\n開始訓練 XGBoost 模型...")
print("   使用網格搜索優化參數...")

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    eval_metric='logloss'
)

grid_search = GridSearchCV(
    xgb_model,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 7. 使用最佳參數
best_model = grid_search.best_estimator_
print(f"\n最佳參數: {grid_search.best_params_}")

# 8. 評估
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("\n" + "=" * 80)
print("模型評估結果")
print("=" * 80)
print(f"訓練準確度: {train_acc*100:.2f}%")
print(f"測試準確度: {test_acc*100:.2f}%")
print("\n分類報告:")
print(classification_report(y_test, y_test_pred, target_names=['不買', '買入']))

# 9. 特徵重要性
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n特徵重要性 (Top 10):")
print(feature_importance.head(10).to_string(index=False))

# 10. 保存模型
model_filename = 'xgb_amzn_model.pkl'
joblib.dump(best_model, model_filename)
print(f"\n模型已保存: {model_filename}")

# 11. 更新準確度數據
accuracy_data = {
    'symbol': 'AMZN',
    'model_type': 'XGBoost',
    'training_accuracy': float(train_acc * 100),
    'validation_accuracy': float(test_acc * 100),
    'backtest_accuracy': float(test_acc * 100),
    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'best_params': grid_search.best_params_
}

with open('model_accuracy_AMZN.json', 'w', encoding='utf-8') as f:
    json.dump(accuracy_data, f, ensure_ascii=False, indent=2)

print(f"準確度數據已更新: model_accuracy_AMZN.json")

# 12. 保存特徵重要性
feature_importance_data = {
    'ticker': 'AMZN',
    'analysis_date': datetime.now().strftime('%Y-%m-%d'),
    'model_type': 'XGBoost',
    'model_accuracy': float(test_acc),
    'feature_importance': {
        row['feature']: float(row['importance'])
        for _, row in feature_importance.iterrows()
    }
}

with open('AMZN_feature_importance.json', 'w', encoding='utf-8') as f:
    json.dump(feature_importance_data, f, ensure_ascii=False, indent=2)

print(f"特徵重要性已保存: AMZN_feature_importance.json")

print("\n" + "=" * 80)
print("訓練完成！")
print("=" * 80)
print(f"模型類型: XGBoost")
print(f"測試準確度: {test_acc*100:.2f}%")
print(f"模型文件: {model_filename}")
print("=" * 80)
