"""
8021.TW (尖點科技) - XGBoost 分類模型
XGBoost 預測明日漲跌方向 (二元分類)
結果保存為 xgb_8021_compare.pkl
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime
import json

def make_features(df):
    df = df.copy()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    d = df['close'].diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + g / (l + 1e-10)))
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + df['bb_std'] * 2
    df['bb_lower'] = df['bb_middle'] - df['bb_std'] * 2
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['price_change_1'] = df['close'].pct_change(1)
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_change_10'] = df['close'].pct_change(10)
    df['volume_ma'] = df['volume'].rolling(10).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
    df['high_low_range'] = (df['high'] - df['low']) / df['close']

    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    features = ['sma_10', 'sma_30', 'sma_50', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_pct', 'price_change_1', 'price_change_5', 'price_change_10',
                'volume_ratio', 'high_low_range']

    df = df.dropna()
    X = df[features].values
    y = df['target'].values
    return X, y, df

if __name__ == '__main__':
    print("=" * 70)
    print("XGBoost 訓練 8021.TW (尖點科技)")
    print("=" * 70)

    df = yf.download('8021.TW', start='2015-01-01', end='2026-12-31', progress=False)
    if df.empty:
        print("❌ 無數據"); exit(1)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.rename(columns={'Close': 'close', 'Volume': 'volume',
                            'Open': 'open', 'High': 'high', 'Low': 'low'}).reset_index()
    print(f"  {len(df)} 天數據")

    X, y, df_feat = make_features(df)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"  訓練集: {len(X_train)} | 測試集: {len(X_test)}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        verbosity=0,
        random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
    test_acc = accuracy_score(y_test, model.predict(X_test)) * 100
    print(f"  訓練準確度: {train_acc:.2f}%")
    print(f"  測試準確度: {test_acc:.2f}%")

    # Backtest
    test_df = df_feat.iloc[split:].copy()
    test_preds = model.predict(X_test)
    capital = 10000.0
    shares = 0
    for i, pred in enumerate(test_preds):
        price = test_df.iloc[i]['close']
        if pred == 1 and capital > price:
            s = int(capital / price)
            capital -= s * price; shares += s
        elif pred == 0 and shares > 0:
            capital += shares * price; shares = 0
    final = capital + shares * test_df.iloc[-1]['close']
    ret = (final - 10000) / 10000 * 100
    print(f"  回測回報率: {ret:.2f}%  最終價值: ${final:.2f}")

    fname = 'xgb_8021_compare.pkl'
    joblib.dump(model, fname)
    print(f"  ✅ 保存: {fname}")

    with open('xgb_accuracy_8021.json', 'w', encoding='utf-8') as f:
        json.dump({'symbol': '8021', 'model_type': 'XGBoost',
                   'train_accuracy': float(train_acc),
                   'test_accuracy': float(test_acc),
                   'backtest_return': float(ret),
                   'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                  f, ensure_ascii=False, indent=2)

    print(f"\n完成！測試準確度 {test_acc:.1f}% | 回報 {ret:.1f}%")
