"""
STX, SNPS - XGBoost 分類模型 (與 SAC 比較)
XGBoost 預測明日漲跌方向 (二元分類)
結果保存為 xgb_stx_compare.pkl / xgb_snps_compare.pkl
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime
import json

STOCKS = [
    ('STX',  'stx',  'Seagate'),
    ('SNPS', 'snps', 'Synopsys'),
]

def make_features(df):
    """Build feature matrix from OHLCV + technical indicators"""
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

    # Target: will tomorrow's close be higher than today's?
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    features = ['sma_10', 'sma_30', 'sma_50', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_pct', 'price_change_1', 'price_change_5', 'price_change_10',
                'volume_ratio', 'high_low_range']

    df = df.dropna()
    X = df[features].values
    y = df['target'].values
    return X, y, df

def train(ticker, symbol, name):
    print(f"\n{'='*70}\nXGBoost 訓練 {ticker} ({name})\n{'='*70}")
    try:
        df = yf.download(ticker, start='2015-01-01', end='2026-12-31', progress=False)
        if df.empty: print("  ❌ 無數據"); return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df = df.rename(columns={'Close': 'close', 'Volume': 'volume',
                                'Open': 'open', 'High': 'high', 'Low': 'low'}).reset_index()
        print(f"  {len(df)} 天數據")

        X, y, df_feat = make_features(df)

        # Split: 80% train, 20% test (time-based)
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
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        # Accuracy
        train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
        test_acc = accuracy_score(y_test, model.predict(X_test)) * 100
        print(f"  訓練準確度: {train_acc:.2f}%")
        print(f"  測試準確度: {test_acc:.2f}%")

        # Simple backtest: buy when model predicts up, sell/hold when down
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

        fname = f'xgb_{symbol}_compare.pkl'
        joblib.dump(model, fname)
        print(f"  ✅ 保存: {fname}")

        with open(f'xgb_accuracy_{symbol}.json', 'w', encoding='utf-8') as f:
            json.dump({'symbol': symbol, 'model_type': 'XGBoost',
                       'train_accuracy': float(train_acc),
                       'test_accuracy': float(test_acc),
                       'backtest_return': float(ret),
                       'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                      f, ensure_ascii=False, indent=2)
        return test_acc, ret
    except Exception as e:
        print(f"  ❌ {e}"); import traceback; traceback.print_exc(); return None


if __name__ == '__main__':
    print("=" * 70)
    print("XGBoost vs SAC 比較訓練 — STX + SNPS")
    print("=" * 70)
    results = []
    for t, s, n in STOCKS:
        r = train(t, s, n)
        results.append((t, n, r))

    print("\n=== XGBoost 結果摘要 ===")
    for t, n, r in results:
        if r:
            print(f"  {t} ({n}): 測試準確度={r[0]:.1f}%  回報={r[1]:.1f}%")
        else:
            print(f"  {t} ({n}): 失敗")
    print("\n完成後與 SAC 結果比較")
