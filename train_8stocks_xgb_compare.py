"""
批量訓練 8 支台灣股票 XGBoost 模型，並與 PPO 結果比較
Stocks: 2451, 3715, 2345, 3131, 3653, 8021, 2449, 2382
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

STOCKS = [
    ('2451.TW', '2451', '創見資訊'),
    ('3715.TW', '3715', '定穎投控'),
    ('2345.TW', '2345', '智邦科技'),
    ('3131.TWO', '3131', '弘塑科技'),   # .TW fails, use .TWO
    ('3653.TW', '3653', '健策精密'),
    ('8021.TW', '8021', '尖點科技'),
    ('2449.TW', '2449', '京元電子'),
    ('2382.TW', '2382', '廣達電腦'),
]

# PPO results from previous training
PPO_RESULTS = {
    '2451': {'acc': 100.00, 'ret': 462.64,   'trades': 6},
    '3715': {'acc': 51.05,  'ret': 927.32,   'trades': 50},
    '2345': {'acc': 51.60,  'ret': 14204.43, 'trades': 39},
    '3131': {'acc': 51.07,  'ret': 3518.22,  'trades': 395},
    '3653': {'acc': 48.38,  'ret': 7501.95,  'trades': 385},
    '8021': {'acc': 0.00,   'ret': 0.00,     'trades': 0},
    '2449': {'acc': 49.10,  'ret': 1990.58,  'trades': 32},
    '2382': {'acc': 100.00, 'ret': 412.66,   'trades': 5},
}

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
    return df[features].values, df['target'].values, df


def train(ticker, symbol, name):
    print(f"\n{'='*70}")
    print(f"XGBoost 訓練 {ticker} ({name})")
    print(f"{'='*70}")
    try:
        df = yf.download(ticker, start='2015-01-01', end='2026-12-31', progress=False)
        if df.empty:
            print(f"  ❌ 無數據"); return None
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

        test_df = df_feat.iloc[split:].copy()
        test_preds = model.predict(X_test)
        capital = 10000.0; shares = 0
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

        return {'acc': test_acc, 'ret': ret, 'train_acc': train_acc}

    except Exception as e:
        print(f"  ❌ {e}")
        import traceback; traceback.print_exc()
        return None


if __name__ == '__main__':
    print("=" * 70)
    print("批量訓練 8 支台灣股票 XGBoost 模型 (對比 PPO)")
    print(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    xgb_results = {}
    for ticker, symbol, name in STOCKS:
        r = train(ticker, symbol, name)
        xgb_results[symbol] = r

    # Comparison table
    print(f"\n\n{'='*90}")
    print("📊 PPO vs XGBoost 比較結果")
    print(f"{'='*90}")
    print(f"{'股票':<8} {'名稱':<10} {'PPO準確度':>10} {'PPO回報':>12} {'PPO交易':>8} {'XGB準確度':>10} {'XGB回報':>12} {'勝出'}")
    print("-" * 90)

    for _, symbol, name in STOCKS:
        ppo = PPO_RESULTS.get(symbol, {})
        xgb = xgb_results.get(symbol)

        ppo_acc = ppo.get('acc', 0)
        ppo_ret = ppo.get('ret', 0)
        ppo_trades = ppo.get('trades', 0)

        if xgb:
            xgb_acc = xgb['acc']
            xgb_ret = xgb['ret']
            # Winner: higher test accuracy (excluding 100% PPO = buy-and-hold)
            if ppo_acc == 100.0 and ppo_trades <= 6:
                winner = "XGB✓ (PPO=buy&hold)"
            elif xgb_acc > ppo_acc:
                winner = "XGB✓"
            elif ppo_acc > xgb_acc:
                winner = "PPO✓"
            else:
                winner = "="
            print(f"{symbol:<8} {name:<10} {ppo_acc:>9.2f}% {ppo_ret:>+11.2f}% {ppo_trades:>8} {xgb_acc:>9.2f}% {xgb_ret:>+11.2f}% {winner}")
        else:
            print(f"{symbol:<8} {name:<10} {ppo_acc:>9.2f}% {ppo_ret:>+11.2f}% {ppo_trades:>8} {'失敗':>10} {'N/A':>12}")

    print(f"\n{'='*90}")
    print("說明: PPO準確度100%+少交易次數 = buy-and-hold行為，非真實準確度")
    print("XGB準確度 = 測試集準確度 (更可靠的評估)")
