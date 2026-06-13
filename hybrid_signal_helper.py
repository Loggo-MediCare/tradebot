"""
Hybrid RF→PPO Signal Helper
============================
Loads hybrid_[code]_[sfx]_ppo.zip models and runs inference.
Observation space: 17-dim
  [0-14]: standard 15 features (shares, balance, price, sma10, sma30, sma50,
           rsi, macd, macd_signal, bb_upper, bb_lower, volume, profit,
           stock_ratio, cash_ratio)
  [15]:   XGBoost buy probability (0-1) as market regime signal
  [16]:   Price vs MA200 ratio (price/MA200 - 1), normalized trend indicator
"""
import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

def find_hybrid_model(code: str) -> str | None:
    """Find hybrid model file for a stock code."""
    code_low = code.lower()
    for sfx in ['tw', 'two', 't', 'hk', 'ks', '']:
        if sfx:
            path = f'hybrid_{code_low}_{sfx}_ppo'
        else:
            path = f'hybrid_{code_low}_ppo'
        if os.path.exists(f'{path}.zip'):
            return path
    return None


def build_obs_17(df: pd.DataFrame, xgb_model=None, initial_balance=10_000.0) -> np.ndarray:
    """
    Build 17-dim observation from current market data.
    Uses last row of df (most recent trading day).
    """
    latest = df.iloc[-1]
    price = float(latest.get('close', latest.get('Close', 0)))

    # Compute technical indicators if not pre-computed
    close = df['close'] if 'close' in df.columns else df['Close']
    volume = df['volume'] if 'volume' in df.columns else df.get('Volume', pd.Series([0]))

    sma10  = float(close.rolling(10).mean().iloc[-1]) if len(close) >= 10 else price
    sma30  = float(close.rolling(30).mean().iloc[-1]) if len(close) >= 30 else price
    sma50  = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else price
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else price

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line = float((ema12 - ema26).iloc[-1])
    macd_sig  = float((ema12 - ema26).ewm(span=9).mean().iloc[-1])

    bb_m = close.rolling(20).mean()
    bb_s = close.rolling(20).std()
    bb_u = float((bb_m + 2*bb_s).iloc[-1]) if len(close) >= 20 else price * 1.05
    bb_l = float((bb_m - 2*bb_s).iloc[-1]) if len(close) >= 20 else price * 0.95

    current_vol = float(volume.iloc[-1]) if len(volume) > 0 else 0

    # OBV as trend proxy
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    obv_now = float(obv.iloc[-1])
    obv_ma  = float(obv.rolling(20).mean().iloc[-1]) if len(obv) >= 20 else 0

    # Dim 15: XGBoost buy probability (regime signal)
    xgb_prob = 0.5  # neutral default
    if xgb_model is not None:
        try:
            feat_cols = ['rsi','macd','macd_signal','macd_hist','bb_position','K','D',
                         'obv','obv_ma20','sma_10','sma_30','sma_50','sma_200',
                         'volatility','atr','price_change_5d','price_change_10d',
                         'price_change_20d','ma50_slope']
            if all(c in df.columns for c in feat_cols):
                X = df[feat_cols].iloc[[-1]]
                xgb_prob = float(xgb_model.predict_proba(X)[0][1])
        except Exception:
            pass

    # Dim 16: price vs MA200 (trend strength)
    ma200_ratio = (price - sma200) / (sma200 + 1e-10)
    ma200_ratio = max(-1.0, min(2.0, ma200_ratio))  # clip

    # Standard 15-dim observation (simulate neutral portfolio state)
    tv = initial_balance  # assume no position for signal
    obs = np.array([
        0.0,                    # shares_held = 0 (fresh signal)
        initial_balance,        # balance
        price,                  # current price
        sma10, sma30, sma50,
        50.0,                   # RSI neutral default (compute below if possible)
        macd_line, macd_sig,
        bb_u, bb_l,
        current_vol,
        0.0,                    # profit = 0
        0.0,                    # stock_ratio = 0
        1.0,                    # cash_ratio = 1
        # Extra dims
        xgb_prob,               # dim 15: XGBoost regime probability
        ma200_ratio,            # dim 16: MA200 trend ratio
    ], dtype=np.float32)

    # Compute RSI if possible
    try:
        d = close.diff()
        g = d.where(d > 0, 0).rolling(14).mean()
        l = (-d.where(d < 0, 0)).rolling(14).mean()
        rsi = float((100 - (100 / (1 + g / (l + 1e-10)))).iloc[-1])
        obs[6] = rsi if not np.isnan(rsi) else 50.0
    except Exception:
        pass

    return obs


def get_hybrid_signal(code: str, df: pd.DataFrame, xgb_model=None) -> dict:
    """
    Run hybrid PPO inference for a stock.
    Returns dict with signal, action_value, model_path.
    """
    result = {'signal': 'N/A', 'action': 0.0, 'model': None, 'available': False}

    model_path = find_hybrid_model(code)
    if not model_path:
        return result

    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        obs = build_obs_17(df, xgb_model)

        # Ensure correct shape
        if obs.shape[0] != model.observation_space.shape[0]:
            # Pad or trim
            target = model.observation_space.shape[0]
            if obs.shape[0] < target:
                obs = np.pad(obs, (0, target - obs.shape[0]))
            else:
                obs = obs[:target]

        action, _ = model.predict(obs, deterministic=True)
        action_val = float(action[0]) if isinstance(action, np.ndarray) else float(action)

        if action_val > 0.1:
            signal = 'BUY'
        elif action_val < -0.1:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        result.update({
            'signal': signal,
            'action': round(action_val, 3),
            'model': model_path,
            'available': True,
        })
    except Exception as e:
        result['error'] = str(e)

    return result


def scan_hybrid_signals(tw_codes: list, period: str = '300d') -> pd.DataFrame:
    """
    Scan hybrid signals for a list of TW stock codes.
    Returns DataFrame with results.
    """
    import yfinance as yf

    rows = []
    total = len(tw_codes)

    for i, (code, sfx) in enumerate(tw_codes, 1):
        ticker = f'{code}.{sfx}'
        print(f'\r  [{i}/{total}] {code}...', end='', flush=True)

        model_path = find_hybrid_model(code)
        if not model_path:
            rows.append({'code': code, 'ticker': ticker, 'signal': '—',
                         'action': 0, 'model': None, 'has_hybrid': False})
            continue

        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if df.empty or len(df) < 30:
                rows.append({'code': code, 'ticker': ticker, 'signal': 'NO DATA',
                             'action': 0, 'model': model_path, 'has_hybrid': True})
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df = df.rename(columns={'Close':'close','Volume':'volume',
                                     'Open':'open','High':'high','Low':'low'})

            res = get_hybrid_signal(code, df)
            price = float(df['close'].iloc[-1])
            rows.append({
                'code': code, 'ticker': ticker,
                'price': round(price, 2),
                'signal': res['signal'],
                'action': res['action'],
                'model': model_path,
                'has_hybrid': True,
            })
        except Exception as e:
            rows.append({'code': code, 'ticker': ticker, 'signal': 'ERROR',
                         'action': 0, 'model': model_path, 'has_hybrid': True,
                         'error': str(e)})

    print()
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # Quick test on 2330
    import yfinance as yf
    df = yf.download('2330.TW', period='300d', progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    df = df.rename(columns={'Close':'close','Volume':'volume','Open':'open','High':'high','Low':'low'})

    res = get_hybrid_signal('2330', df)
    print(f"\n2330 台積電 Hybrid PPO Signal:")
    print(f"  Model:   {res['model']}")
    print(f"  Action:  {res['action']:+.3f}")
    print(f"  Signal:  {res['signal']}")
    print(f"  Available: {res['available']}")
