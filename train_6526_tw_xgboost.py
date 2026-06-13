"""
6526.TW (達發) XGBoost 訓練
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
import json
from datetime import datetime

TICKER = '6526.TW'
print(f"訓練 {TICKER} (達發) XGBoost 模型")
print("=" * 80)

try:
    # 下載數據
    df = yf.download(TICKER, start='2015-01-01', end='2025-12-31', progress=False)

    if df.empty:
        print("ERROR: 無法下載數據")
        sys.exit(1)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df.rename(columns={'Close': 'close', 'Volume': 'volume',
                            'Open': 'open', 'High': 'high', 'Low': 'low'}).reset_index()

    print(f"下載 {len(df)} 天數據")

    # 添加技術指標
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100).fillna(50)

    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['K'] = ((df['close'] - low_14) / (high_14 - low_14) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()

    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma20'] = df['obv'].rolling(20).mean()

    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()

    df['price_change_5d'] = df['close'].pct_change(5) * 100
    df['price_change_10d'] = df['close'].pct_change(10) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100

    df['ma50_slope'] = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100

    # 目標：預測未來5天漲幅 > 2%
    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df['target'] = (df['future_return'] > 0.02).astype(int)

    feature_columns = [
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_position', 'K', 'D', 'obv', 'obv_ma20',
        'sma_10', 'sma_30', 'sma_50', 'sma_200',
        'volatility', 'atr',
        'price_change_5d', 'price_change_10d', 'price_change_20d',
        'ma50_slope'
    ]

    df_clean = df.dropna(subset=feature_columns + ['target'])
    print(f"清理後數據: {len(df_clean)} 天")

    if len(df_clean) < 100:
        print("ERROR: 數據不足 (少於100天)")
        sys.exit(1)

    X = df_clean[feature_columns]
    y = df_clean['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print(f"訓練集: {len(X_train)} 天")
    print(f"測試集: {len(X_test)} 天")

    # 訓練 XGBoost
    model = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=200,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss'
    )

    print("\n開始訓練...")
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"\n訓練準確度: {train_acc*100:.2f}%")
    print(f"測試準確度: {test_acc*100:.2f}%")

    # 保存模型
    joblib.dump(model, 'xgb_6526_tw_model.pkl')
    print(f"\n模型已保存: xgb_6526_tw_model.pkl")

    # 保存準確度
    accuracy_data = {
        'symbol': '6526.TW',
        'model_type': 'XGBoost',
        'training_accuracy': float(train_acc * 100),
        'validation_accuracy': float(test_acc * 100),
        'backtest_accuracy': float(test_acc * 100),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open('model_accuracy_6526_TW.json', 'w', encoding='utf-8') as f:
        json.dump(accuracy_data, f, ensure_ascii=False, indent=2)

    # 特徵重要性
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n特徵重要性:")
    print(feature_importance.head(10))

    fi_data = {
        'ticker': '6526.TW',
        'model_type': 'XGBoost',
        'model_accuracy': float(test_acc),
        'feature_importance': {
            row['feature']: float(row['importance'])
            for _, row in feature_importance.iterrows()
        }
    }

    with open('6526_TW_feature_importance.json', 'w', encoding='utf-8') as f:
        json.dump(fi_data, f, ensure_ascii=False, indent=2)

    status = 'PASS' if test_acc >= 0.58 else 'OK' if test_acc >= 0.50 else 'LOW'
    print(f"\n完成! 狀態: {status} ({test_acc*100:.2f}%)")

except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
