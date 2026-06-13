"""
XGBoost trading signal model trainer.
Trains on technical indicators → predicts BUY/HOLD/SELL.
Records ROI and accuracy in ModelAccuracyTracker.
Usage: python _train_xgboost.py TICKER [TICKER2 ...]
"""
import os, sys, io, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['MPLBACKEND'] = 'Agg'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def download_data(ticker, days=1095):
    end   = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.rename(columns={'Close':'close','Open':'open','High':'high','Low':'low','Volume':'volume'})
    return df.reset_index()


def add_features(df):
    c = df['close']
    # Moving averages
    for w in [5, 10, 20, 50]:
        df[f'sma_{w}'] = c.rolling(w).mean()
        df[f'ema_{w}'] = c.ewm(span=w, adjust=False).mean()
    # RSI
    delta = c.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    # Bollinger Bands
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df['bb_upper']    = sma20 + std20 * 2
    df['bb_lower']    = sma20 - std20 * 2
    df['bb_position'] = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
    # Volume
    df['vol_ratio']   = df['volume'] / df['volume'].rolling(20).mean()
    # Momentum
    for p in [5, 10, 20]:
        df[f'roc_{p}'] = c.pct_change(p) * 100
    # Volatility
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    return df.dropna()


def create_labels(df, forward=5, threshold=0.02):
    """Label: 1=BUY (>+2% in 5 days), -1=SELL (<-2%), 0=HOLD"""
    future_ret = df['close'].shift(-forward) / df['close'] - 1
    labels = np.where(future_ret > threshold, 1,
             np.where(future_ret < -threshold, -1, 0))
    return labels


def backtest_xgb(df, model, scaler, feature_cols, initial_balance=10000):
    """Simple backtest: follow XGBoost signals."""
    balance = initial_balance
    shares  = 0
    trades  = 0

    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)  # 1=BUY, -1=SELL, 0=HOLD (mapped from 0,1,2)
    # Map back: class 0→-1(SELL), class 1→0(HOLD), class 2→1(BUY)
    action_map = {0: -1, 1: 0, 2: 1}
    actions = [action_map[int(p)] for p in preds]

    for i, action in enumerate(actions):
        price = float(df['close'].iloc[i])
        if action == 1 and balance >= price:        # BUY
            buy = int(balance * 0.5 // price)
            if buy > 0:
                balance -= buy * price
                shares  += buy
                trades  += 1
        elif action == -1 and shares > 0:           # SELL
            sell = int(shares * 0.5)
            if sell > 0:
                balance += sell * price
                shares  -= sell
                trades  += 1

    # Liquidate
    final_price = float(df['close'].iloc[-1])
    balance += shares * final_price
    roi = (balance - initial_balance) / initial_balance * 100
    return roi, trades, balance


def train_xgboost(ticker):
    print(f'\n{"="*60}')
    print(f'[XGBoost] Training {ticker}')
    print(f'{"="*60}')

    # 1. Data
    print('Downloading data...')
    df = download_data(ticker)
    df = add_features(df)
    print(f'Data: {len(df)} rows')

    # 2. Features & labels
    feature_cols = [c for c in df.columns if c not in
                    ['Date','open','high','low','close','volume','Adj Close']]
    labels_raw = create_labels(df, forward=5, threshold=0.02)
    df = df.iloc[:len(labels_raw)].copy()
    df['label'] = labels_raw
    df = df.dropna(subset=['label'])
    df = df[df['label'].isin([-1, 0, 1])]

    # Map labels to 0,1,2 for XGBoost
    label_map = {-1: 0, 0: 1, 1: 2}
    y = df['label'].map(label_map).values
    X = df[feature_cols].values

    # 3. Split & scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 4. Train XGBoost
    print('Training XGBoost...')
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,           # shallower = less overfit
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,         # L1 regularization
        reg_lambda=1.5,        # L2 regularization
        min_child_weight=5,    # prevent small leaf splits
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0,
        num_class=3,
        objective='multi:softmax',
        early_stopping_rounds=20,
    )
    model.fit(X_train_s, y_train,
              eval_set=[(X_test_s, y_test)],
              verbose=False)

    # 5. Accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train_s)) * 100
    test_acc  = accuracy_score(y_test,  model.predict(X_test_s))  * 100
    print(f'Train acc: {train_acc:.1f}%  Test acc: {test_acc:.1f}%')

    # 6. Backtest on TEST set only (out-of-sample)
    test_df = df.iloc[len(X_train):].reset_index(drop=True)
    roi, trades, final_bal = backtest_xgb(test_df, model, scaler, feature_cols)
    win_rate = min(100, max(0, 50 + roi / 2))
    from roi_control import print_roi
    print_roi(f'Backtest ROI (test set): {roi:.2f}%  Trades: {trades}  Final: ${final_bal:,.2f}')

    # 7. Save model + scaler
    model_path  = os.path.join(BASE_DIR, f'xgb_{ticker.lower().replace(".", "_")}_model.json')
    scaler_path = os.path.join(BASE_DIR, f'xgb_{ticker.lower().replace(".", "_")}_scaler.pkl')
    model.save_model(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, scaler_path.replace('_scaler.pkl', '_features.pkl'))
    print(f'Model saved: {model_path}')

    # 8. Record accuracy
    sys.path.insert(0, BASE_DIR)
    from model_accuracy_tracker import ModelAccuracyTracker
    tracker = ModelAccuracyTracker(ticker, 'XGBoost')
    tracker.update_training_stats(
        training_acc=train_acc,
        backtest_acc=ModelAccuracyTracker.roi_to_score(roi),
        win_rate=win_rate,
    )
    from roi_control import print_roi
    print_roi(f'Accuracy recorded: backtest_acc={50+roi:.1f}  win_rate={win_rate:.1f}')
    return roi, test_acc


if __name__ == '__main__':
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ['MU']
    results = {}
    for ticker in tickers:
        try:
            roi, acc = train_xgboost(ticker.upper())
            results[ticker] = {'roi': roi, 'acc': acc}
        except Exception as e:
            print(f'ERROR {ticker}: {e}')
            results[ticker] = {'roi': None, 'acc': None}

    print(f'\n{"="*60}')
    print('XGBoost Training Summary')
    print(f'{"="*60}')
    for t, r in results.items():
        if r['roi'] is not None:
            print(f'  {t:10s}  ROI: {r["roi"]:+7.2f}%  Test Acc: {r["acc"]:.1f}%')
        else:
            print(f'  {t:10s}  FAILED')
