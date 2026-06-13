"""
台股 3013 晟鈦 AI 交易信号生成器
使用训练好的 XGBoost 模型生成今日交易策略
"""
import os, sys, io, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime

TICKER   = '3013.TW'
NAME     = '3013 晟鈦'
MODEL    = 'xgb_3013_tw_model.pkl'
FEATURE_COLUMNS = [
    'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position', 'K', 'D',
    'obv', 'obv_ma20', 'sma_10', 'sma_30', 'sma_50', 'sma_200',
    'volatility', 'atr', 'price_change_5d', 'price_change_10d', 'price_change_20d', 'ma50_slope'
]
sep = '=' * 60


def add_technical_indicators(df):
    df['sma_10']  = df['close'].rolling(10).mean()
    df['sma_30']  = df['close'].rolling(30).mean()
    df['sma_50']  = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['ema_12']  = df['close'].ewm(span=12).mean()
    df['ema_26']  = df['close'].ewm(span=26).mean()
    d = df['close'].diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + g / (l + 1e-10)))
    df['macd']        = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    df['bb_middle']   = df['close'].rolling(20).mean()
    df['bb_std']      = df['close'].rolling(20).std()
    df['bb_upper']    = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower']    = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100).fillna(50)
    lo14 = df['low'].rolling(14).min()
    hi14 = df['high'].rolling(14).max()
    df['K'] = ((df['close'] - lo14) / (hi14 - lo14) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()
    df['obv']      = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma20'] = df['obv'].rolling(20).mean()
    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    hl = df['high'] - df['low']
    hc = np.abs(df['high'] - df['close'].shift())
    lc = np.abs(df['low']  - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['price_change_5d']  = df['close'].pct_change(5)  * 100
    df['price_change_10d'] = df['close'].pct_change(10) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100
    df['ma50_slope'] = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100
    return df.bfill().ffill()


def get_trading_signal():
    df_raw = yf.download(TICKER, period='1y', progress=False, auto_adjust=True)
    if df_raw.empty:
        print('  No data for', TICKER)
        return None

    df = df_raw.copy()
    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]

    df = add_technical_indicators(df)
    if len(df) < 10:
        print('  Not enough data')
        return None

    model = joblib.load(MODEL)
    row   = df.iloc[-1]
    price = float(row['close'])
    rsi   = float(row['rsi'])
    macd  = float(row['macd'])
    feat  = row[FEATURE_COLUMNS].values.reshape(1, -1)

    prob = float(model.predict_proba(feat)[0][1])
    pred = int(model.predict(feat)[0])

    if pred == 1 and prob >= 0.55 and rsi < 75:
        signal_ch = '🟢 買入信號 (BUY)'
    elif pred == 0 and prob < 0.45 and rsi > 55:
        signal_ch = '🔴 賣出信號 (SELL)'
    else:
        signal_ch = '🟡 持有觀望 (HOLD)'

    print(f'\n{sep}')
    print(f'  {NAME}  交易信號')
    print(sep)
    print(f'  日期    : {datetime.now().strftime("%Y-%m-%d")}')
    print(f'  現價    : NT${price:.2f}')
    print(f'  RSI(14) : {rsi:.1f}')
    print(f'  MACD    : {macd:.4f}')
    print(f'  買入概率: {prob*100:.1f}%')
    print(f'  信號    : {signal_ch}')
    print(sep)
    return {'signal': signal_ch, 'price': price, 'prob': prob, 'rsi': rsi}


if __name__ == '__main__':
    get_trading_signal()
