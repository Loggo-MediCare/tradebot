import sys; sys.argv=['_train_hybrid_rf_ppo.py','2884.TW']
"""
Model 4: Hybrid RF→PPO
======================
Step 1: Train Random Forest → generates BUY/SELL/HOLD probabilities per day
Step 2: Feed RF probabilities as extra features into PPO observation space
Step 3: PPO trains with enhanced state (tech indicators + RF signal)
Step 4: Backtest and record ROI in ModelAccuracyTracker

Usage: python _train_hybrid_rf_ppo.py TICKER [TICKER2 ...]
"""
import os, sys, io, warnings, tempfile, subprocess, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MPLBACKEND'] = 'Agg'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime, timedelta
import yfinance as yf
from roi_control import print_roi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Data & Features ──────────────────────────────────────────────────────────

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


def add_indicators(df):
    c = df['close']
    for w in [10, 20, 30, 50]:
        df[f'sma_{w}'] = c.rolling(w).mean()
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    delta = c.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi']         = 100 - (100 / (1 + gain / (loss + 1e-9)))
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df['bb_upper']    = sma20 + std20 * 2
    df['bb_lower']    = sma20 - std20 * 2
    df['bb_position'] = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
    df['vol_ratio']   = df['volume'] / df['volume'].rolling(20).mean()
    for p in [5, 10, 20]:
        df[f'roc_{p}'] = c.pct_change(p) * 100
    df['atr']         = (df['high'] - df['low']).rolling(14).mean()
    return df.dropna()


FEATURE_COLS = ['sma_10','sma_20','sma_30','sma_50','rsi','macd','macd_signal',
                'bb_position','vol_ratio','roc_5','roc_10','roc_20','atr']


# ── Step 1: Random Forest ─────────────────────────────────────────────────────

def train_random_forest(df, forward=5, threshold=0.02):
    """Train RF classifier: labels = next-5-day direction."""
    future_ret = df['close'].shift(-forward) / df['close'] - 1
    labels = np.where(future_ret > threshold, 2,       # BUY
             np.where(future_ret < -threshold, 0, 1))  # SELL / HOLD

    valid = ~np.isnan(future_ret.values)
    X = df[FEATURE_COLS].values[valid]
    y = labels[valid]

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=6,
        min_samples_leaf=5, random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_s, y)

    # Feature importance
    importance = dict(zip(FEATURE_COLS, rf.feature_importances_))
    top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f'  RF top features: {[(k, f"{v:.3f}") for k,v in top]}')
    acc = accuracy_score(y, rf.predict(X_s))
    print(f'  RF train accuracy: {acc:.1%}')

    return rf, scaler, valid


def get_rf_probabilities(df, rf, scaler, valid):
    """Get RF signal probabilities for each day. Returns array (N, 3)."""
    X     = df[FEATURE_COLS].values[valid]
    X_s   = scaler.transform(X)
    proba = rf.predict_proba(X_s)   # shape (N, 3): [P(SELL), P(HOLD), P(BUY)]
    return proba                     # classes: 0=SELL, 1=HOLD, 2=BUY


# ── Step 2: Hybrid Trading Environment ───────────────────────────────────────

class HybridTradingEnv(gym.Env):
    """PPO env with RF signal probabilities as extra observation features."""

    def __init__(self, df, rf_proba, initial_balance=10000):
        super().__init__()
        self.df              = df.reset_index(drop=True)
        self.rf_proba        = rf_proba        # shape (len(df), 3)
        self.initial_balance = initial_balance
        # Obs: [shares, balance, close, sma10, sma30, rsi, macd, macd_sig,
        #       bb_upper, bb_lower, volume, profit, stock_ratio, cash_ratio,
        #       rf_p_sell, rf_p_hold, rf_p_buy]  → 17 features
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance      = self.initial_balance
        self.shares       = 0
        self.profit       = 0
        self.trades       = 0
        return self._obs(), {}

    def _obs(self):
        row   = self.df.iloc[self.current_step]
        price = float(row['close'])
        tv    = self.balance + self.shares * price
        sr    = (self.shares * price) / tv if tv > 0 else 0
        cr    = self.balance / tv if tv > 0 else 1
        rf    = self.rf_proba[self.current_step]   # [p_sell, p_hold, p_buy]
        return np.array([
            float(self.shares), float(self.balance), price,
            float(row.get('sma_10', 0)), float(row.get('sma_30', 0)),
            float(row.get('rsi', 50)), float(row.get('macd', 0)),
            float(row.get('macd_signal', 0)),
            float(row.get('bb_upper', 0)), float(row.get('bb_lower', 0)),
            float(row.get('volume', 0)), float(self.profit), sr, cr,
            float(rf[0]), float(rf[1]), float(rf[2]),   # RF probabilities
        ], dtype=np.float32)

    def step(self, action):
        a     = float(action[0]) if hasattr(action, '__len__') else float(action)
        a     = max(-1.0, min(1.0, a))
        price = float(self.df.iloc[self.current_step]['close'])

        # RF signal bonus: if RF strongly says BUY, boost buy; if SELL, boost sell
        rf_buy  = self.rf_proba[self.current_step][2]
        rf_sell = self.rf_proba[self.current_step][0]
        rf_boost = (rf_buy - rf_sell) * 0.2   # small nudge

        if a < -0.1 and self.shares > 0:
            sell = int(self.shares * abs(a))
            if sell > 0:
                self.balance += sell * price; self.shares -= sell; self.trades += 1
        elif a > 0.1 and self.balance >= price:
            buy = int((self.balance * a) // price)
            if buy > 0:
                self.balance -= buy * price; self.shares += buy; self.trades += 1

        self.current_step += 1
        done  = self.current_step >= len(self.df) - 1
        price_now = float(self.df.iloc[self.current_step]['close'])
        tv    = self.balance + self.shares * price_now
        self.profit = tv - self.initial_balance
        reward = (self.profit / self.initial_balance) + rf_boost

        if done:
            self.balance += self.shares * price_now; self.shares = 0
        return self._obs(), float(reward), done, False, {}


# ── Step 3: Train Hybrid PPO ──────────────────────────────────────────────────

def train_hybrid(ticker, timesteps=150000):
    print(f'\n{"="*60}')
    print(f'[Hybrid RF→PPO] Training {ticker}')
    print(f'{"="*60}')

    # Data
    df = download_data(ticker)
    df = add_indicators(df)
    print(f'Data: {len(df)} rows')

    # RF
    print('\nStep 1: Training Random Forest...')
    rf, rf_scaler, valid = train_random_forest(df)
    rf_proba = get_rf_probabilities(df, rf, rf_scaler, valid)
    df_valid = df[valid].reset_index(drop=True)

    # Split 80/20
    split = int(len(df_valid) * 0.8)
    train_df   = df_valid.iloc[:split].reset_index(drop=True)
    test_df    = df_valid.iloc[split:].reset_index(drop=True)
    train_prob = rf_proba[:split]
    test_prob  = rf_proba[split:]

    # PPO
    print(f'\nStep 2: Training PPO ({timesteps:,} steps)...')
    env   = DummyVecEnv([lambda: HybridTradingEnv(train_df, train_prob)])
    model = PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=1024,
                batch_size=64, n_epochs=10, verbose=0)
    model.learn(total_timesteps=timesteps)

    # Save
    model_path = os.path.join(BASE_DIR, f'hybrid_{ticker.lower().replace(".", "_")}_ppo')
    rf_path    = os.path.join(BASE_DIR, f'hybrid_{ticker.lower().replace(".", "_")}_rf.pkl')
    model.save(model_path)
    joblib.dump({'rf': rf, 'scaler': rf_scaler}, rf_path)
    print(f'Model saved: {model_path}.zip')

    # Backtest on test set
    print('\nStep 3: Backtesting on test set...')
    test_env = HybridTradingEnv(test_df, test_prob)
    test_env.portfolio_history = []
    obs, _   = test_env.reset()
    done     = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = test_env.step(action)
        tv = test_env.balance + test_env.shares * float(test_df['close'].iloc[min(test_env.current_step, len(test_df)-1)])
        test_env.portfolio_history.append(tv)

    roi      = test_env.profit / test_env.initial_balance * 100
    win_rate = min(100, max(0, 50 + roi / 3))

    # Sharpe ratio from portfolio history
    import math
    portfolio_vals = getattr(test_env, 'portfolio_history', None)
    sharpe = 0.0
    if portfolio_vals and len(portfolio_vals) > 1:
        pv  = np.array(portfolio_vals)
        dr  = np.diff(pv) / pv[:-1]
        ann = np.mean(dr) * 252 - 0.05
        std = np.std(dr) * math.sqrt(252)
        sharpe = round(ann / std, 4) if std > 0 else 0.0

    print_roi(f'Backtest ROI: {roi:.2f}%  Sharpe: {sharpe:.3f}  Trades: {test_env.trades}')
    print_roi(f'Final Balance: ${test_env.balance:,.2f}')

    # Record accuracy
    from model_accuracy_tracker import ModelAccuracyTracker
    tracker = ModelAccuracyTracker(ticker, 'Hybrid')
    tracker.update_training_stats(
        backtest_acc=ModelAccuracyTracker.roi_to_score(roi),
        win_rate=win_rate,
        sharpe_ratio=sharpe,
    )
    print_roi(f'Accuracy recorded for Hybrid {ticker}: ROI={roi:.2f}%  Sharpe={sharpe:.3f}')
    return roi, sharpe


if __name__ == '__main__':
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ['MU']
    results = {}
    for ticker in tickers:
        try:
            result = train_hybrid(ticker.upper())
            roi, sharpe = result if isinstance(result, tuple) else (result, 0.0)
            results[ticker] = (roi, sharpe)
        except Exception as e:
            print(f'ERROR {ticker}: {e}')
            import traceback; traceback.print_exc()
            results[ticker] = None

    print(f'\n{"="*60}')
    print('Hybrid RF→PPO Training Summary')
    print(f'{"="*60}')
    for t, val in results.items():
        roi, sharpe = val if val else (None, None)
        if roi is not None:
            print(f'  {t:10s}  ROI: {roi:+.2f}%  Sharpe: {sharpe:.3f}')
        else:
            print(f'  {t:10s}  FAILED')
