"""
美股 SHOP Shopify AI 交易信号生成器
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

TICKER   = 'SHOP'
NAME     = 'SHOP Shopify'
MODEL    = 'xgb_shop_model.pkl'
FEATURES = ['sma_10','sma_30','sma_50','rsi','macd','macd_signal',
            'bb_upper','bb_lower','obv','volume_ratio']


def add_features(df_raw):
    df = df_raw.copy()
    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df['sma_10']  = df['close'].rolling(10).mean()
    df['sma_30']  = df['close'].rolling(30).mean()
    df['sma_50']  = df['close'].rolling(50).mean()
    df['ema_12']  = df['close'].ewm(span=12).mean()
    df['ema_26']  = df['close'].ewm(span=26).mean()
    d = df['close'].diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    df['rsi']         = 100 - 100 / (1 + g / (l + 1e-10))
    df['macd']        = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['bb_mid']      = df['close'].rolling(20).mean()
    df['bb_std']      = df['close'].rolling(20).std()
    df['bb_upper']    = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower']    = df['bb_mid'] - 2 * df['bb_std']
    df['obv']          = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['vol_ma20']     = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['vol_ma20'] + 1e-10)
    return df.bfill().ffill().dropna()


def get_trading_signal():
    df_raw = yf.download(TICKER, period='6mo', progress=False, auto_adjust=True)
    if df_raw.empty:
        print('  No data for', TICKER)
        return None

    df = add_features(df_raw)
    if len(df) < 10:
        print('  Not enough data')
        return None

    model = joblib.load(MODEL)
    row   = df.iloc[-1]
    price = float(row['close'])
    rsi   = float(row['rsi'])
    macd  = float(row['macd'])
    feat  = row[FEATURES].values.reshape(1, -1)

    prob = float(model.predict_proba(feat)[0][1])
    pred = int(model.predict(feat)[0])

    if pred == 1 and prob >= 0.55 and rsi < 75:
        signal_ch = '🟢 買入信號 (BUY)'
    elif pred == 0 and prob < 0.45 and rsi > 55:
        signal_ch = '🔴 賣出信號 (SELL)'
    else:
        signal_ch = '🟡 持有觀望 (HOLD)'

    print(f'\n============================================================')
    print(f'  {NAME}  交易信號')
    print('=' * 60)
    print(f'  日期    : {datetime.now().strftime("%Y-%m-%d")}')
    print(f'  現價    : ${price:.2f}')
    print(f'  RSI(14) : {rsi:.1f}')
    print(f'  MACD    : {macd:.4f}')
    print(f'  買入概率: {prob*100:.1f}%')
    print(f'  信號    : {signal_ch}')
    print('=' * 60)
    return {'signal': signal_ch, 'price': price, 'prob': prob, 'rsi': rsi}


if __name__ == '__main__':
    get_trading_signal()
