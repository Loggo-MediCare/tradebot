"""
ROI Backtest: XGBoost vs PPO — 3491.TWO & 3105.TWO
Simulates trading on the test set and reports final ROI, win-rate, trades.
"""
import os, sys, io, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import json
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

STOCKS = [
    {'code': '3491', 'ticker': '3491.TWO', 'name': '昇達科'},
    {'code': '3105', 'ticker': '3105.TWO', 'name': '穩懋'},
]
INITIAL_BALANCE = 100_000   # NT$100,000
END_DATE = '2026-05-01'

FEATURE_COLUMNS = [
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_position', 'K', 'D', 'obv', 'obv_ma20',
    'sma_10', 'sma_30', 'sma_50', 'sma_200',
    'volatility', 'atr',
    'price_change_5d', 'price_change_10d', 'price_change_20d',
    'ma50_slope'
]

def add_indicators(df):
    df['sma_10']  = df['close'].rolling(10).mean()
    df['sma_30']  = df['close'].rolling(30).mean()
    df['sma_50']  = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['ema_12']  = df['close'].ewm(span=12).mean()
    df['ema_26']  = df['close'].ewm(span=26).mean()
    delta = df['close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df['macd']        = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    df['bb_middle']   = df['close'].rolling(20).mean()
    df['bb_std']      = df['close'].rolling(20).std()
    df['bb_upper']    = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower']    = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = ((df['close'] - df['bb_lower']) /
                         (df['bb_upper'] - df['bb_lower']) * 100).fillna(50)
    lo14 = df['low'].rolling(14).min()
    hi14 = df['high'].rolling(14).max()
    df['K'] = ((df['close'] - lo14) / (hi14 - lo14) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()
    df['obv']      = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma20'] = df['obv'].rolling(20).mean()
    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    hl  = df['high'] - df['low']
    hc  = np.abs(df['high'] - df['close'].shift())
    lc  = np.abs(df['low']  - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['price_change_5d']  = df['close'].pct_change(5)  * 100
    df['price_change_10d'] = df['close'].pct_change(10) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100
    df['ma50_slope']       = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100
    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df['target']        = (df['future_return'] > 0.02).astype(int)
    return df.bfill().ffill()


# ─── XGBoost ROI backtest ─────────────────────────────────────────────────────
def backtest_xgboost(model, test_df, initial_balance=INITIAL_BALANCE):
    """
    Signal: predict=1 → BUY all-in; predict=0 → SELL all; hold otherwise.
    """
    balance = initial_balance
    shares  = 0
    trades  = 0
    wins    = 0
    equity_curve = [balance]
    buy_price = None

    X = test_df[FEATURE_COLUMNS]
    preds = model.predict(X)

    for i, (idx, row) in enumerate(test_df.iterrows()):
        price = float(row['close'])
        sig   = int(preds[i])

        if sig == 1 and balance > price:        # BUY
            shares_to_buy = int(balance // price)
            if shares_to_buy > 0:
                balance -= shares_to_buy * price
                shares  += shares_to_buy
                buy_price = price
                trades += 1

        elif sig == 0 and shares > 0:           # SELL
            proceeds = shares * price
            balance += proceeds
            if buy_price and price > buy_price:
                wins += 1
            shares = 0
            buy_price = None

        total = balance + shares * price
        equity_curve.append(total)

    # Force-close at end
    if shares > 0:
        balance += shares * float(test_df.iloc[-1]['close'])
        shares = 0

    final = balance
    roi   = (final - initial_balance) / initial_balance * 100
    win_rate = wins / trades * 100 if trades > 0 else 0
    bh_roi = (float(test_df.iloc[-1]['close']) / float(test_df.iloc[0]['close']) - 1) * 100
    return roi, trades, win_rate, bh_roi, equity_curve


# ─── PPO environment (same as training) ──────────────────────────────────────
class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=INITIAL_BALANCE):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.balance = float(self.initial_balance)
        self.shares = 0
        self.profit = 0.0
        return self._obs(), {}

    def _obs(self):
        r  = self.df.iloc[self.step_idx]
        p  = float(r['close'])
        tv = self.balance + self.shares * p
        return np.array([
            float(self.shares), float(self.balance), p,
            float(r.get('sma_10', 0)), float(r.get('sma_30', 0)),
            float(r.get('sma_50', 0)), float(r.get('rsi', 50)),
            float(r.get('macd', 0)), float(r.get('macd_signal', 0)),
            float(r.get('bb_upper', 0)), float(r.get('bb_lower', 0)),
            float(r.get('volume', 0)), float(self.profit),
            (self.shares * p) / tv if tv > 0 else 0,
            self.balance / tv if tv > 0 else 1,
        ], dtype=np.float32)

    def step(self, action):
        a = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        a = np.clip(a, -1.0, 1.0)
        p = float(self.df.iloc[self.step_idx]['close'])
        if a < -0.1:
            s = int(self.shares * abs(a))
            if s > 0: self.balance += s * p; self.shares -= s
        elif a > 0.1:
            s = int((self.balance // p) * a)
            if s > 0: self.balance -= s * p; self.shares += s
        self.profit = (self.balance + self.shares * p) - self.initial_balance
        self.step_idx += 1
        done = self.step_idx >= len(self.df) - 1
        reward = self.profit / self.initial_balance
        if abs(a) > 0.1: reward += 0.01
        if self.balance > (self.balance + self.shares * p) * 0.9: reward -= 0.005
        return self._obs(), reward, done, False, {}


def backtest_ppo(model, test_df, initial_balance=INITIAL_BALANCE):
    env = TradingEnv(test_df.reset_index(drop=True), initial_balance=initial_balance)
    obs, _ = env.reset()
    trades  = 0
    wins    = 0
    equity_curve = [initial_balance]
    buy_price = None
    last_shares = 0

    for _ in range(len(test_df) - 1):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(act)
        a = float(act[0])

        price = float(env.df.iloc[env.step_idx - 1]['close'])
        # detect trade events
        if a > 0.1 and env.shares > last_shares:   # bought
            buy_price = price
            trades += 1
        elif a < -0.1 and env.shares < last_shares and last_shares > 0:  # sold
            if buy_price and price > buy_price:
                wins += 1
            buy_price = None
        last_shares = env.shares

        total = env.balance + env.shares * price
        equity_curve.append(total)
        if done:
            break

    final = env.balance + env.shares * float(test_df.iloc[-1]['close'])
    roi   = (final - initial_balance) / initial_balance * 100
    win_rate = wins / trades * 100 if trades > 0 else 0
    bh_roi = (float(test_df.iloc[-1]['close']) / float(test_df.iloc[0]['close']) - 1) * 100
    return roi, trades, win_rate, bh_roi, equity_curve


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 72)
    print("  💰 ROI BACKTEST: XGBoost vs PPO — 3491 & 3105")
    print(f"  Initial capital: NT${INITIAL_BALANCE:,}  |  Test set = last 20% of data")
    print("=" * 72)

    all_results = []

    for s in STOCKS:
        code, ticker, name = s['code'], s['ticker'], s['name']
        suffix = 'two' if '.TWO' in ticker else 'tw'

        print(f"\n{'─'*72}")
        print(f"  {ticker} ({name})")
        print(f"{'─'*72}")

        # ── Download & prepare ──
        df = yf.download(ticker, start='2015-01-01', end=END_DATE, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty:
            print(f"  ❌ No data for {ticker}")
            continue
        df = df.rename(columns={'Close':'close','Volume':'volume',
                                'Open':'open','High':'high','Low':'low'}).reset_index()
        df = add_indicators(df)
        df_clean = df.dropna(subset=FEATURE_COLUMNS + ['target'])

        split    = int(len(df_clean) * 0.8)
        test_df  = df_clean.iloc[split:].copy().reset_index(drop=True)
        test_start = str(test_df.iloc[0]['Date'])[:10]
        test_end   = str(test_df.iloc[-1]['Date'])[:10]
        print(f"  Test period: {test_start} → {test_end}  ({len(test_df)} days)")

        # ── XGBoost backtest ──
        xgb_file = f'xgb_{code}_{suffix}_model.pkl'
        try:
            xgb_model = joblib.load(xgb_file)
            xgb_roi, xgb_trades, xgb_wr, bh_roi, xgb_curve = backtest_xgboost(xgb_model, test_df)
            print(f"\n  [XGBoost]")
            print(f"    ROI:       {xgb_roi:+.2f}%")
            print(f"    Trades:    {xgb_trades}")
            print(f"    Win rate:  {xgb_wr:.1f}%")
            print(f"    Final val: NT${INITIAL_BALANCE * (1 + xgb_roi/100):,.0f}")
        except FileNotFoundError:
            print(f"  ❌ XGBoost model not found: {xgb_file}")
            xgb_roi = xgb_trades = xgb_wr = None

        # ── PPO backtest ──
        ppo_file = f'ppo_{code}_{suffix}_improved'
        try:
            ppo_model = PPO.load(ppo_file)
            ppo_roi, ppo_trades, ppo_wr, bh_roi, ppo_curve = backtest_ppo(ppo_model, test_df)
            print(f"\n  [PPO]")
            print(f"    ROI:       {ppo_roi:+.2f}%")
            print(f"    Trades:    {ppo_trades}")
            print(f"    Win rate:  {ppo_wr:.1f}%")
            print(f"    Final val: NT${INITIAL_BALANCE * (1 + ppo_roi/100):,.0f}")
        except Exception as e:
            print(f"  ❌ PPO model error: {e}")
            ppo_roi = ppo_trades = ppo_wr = None

        # Buy-and-hold benchmark
        print(f"\n  [Buy & Hold]")
        print(f"    ROI:       {bh_roi:+.2f}%")
        print(f"    Final val: NT${INITIAL_BALANCE * (1 + bh_roi/100):,.0f}")

        all_results.append({
            'ticker': ticker, 'name': name,
            'xgb_roi': xgb_roi, 'xgb_trades': xgb_trades, 'xgb_wr': xgb_wr,
            'ppo_roi': ppo_roi, 'ppo_trades': ppo_trades, 'ppo_wr': ppo_wr,
            'bh_roi': bh_roi,
            'test_start': test_start, 'test_end': test_end,
        })

    # ── Final summary table ──
    print(f"\n{'='*72}")
    print("  📊 ROI COMPARISON TABLE  (Initial: NT$100,000)")
    print(f"{'='*72}")
    print(f"  {'Ticker':<12} {'Name':<8} {'XGB ROI':>9} {'PPO ROI':>9} {'B&H ROI':>9} {'Best':>10}")
    print(f"  {'─'*62}")
    for r in all_results:
        xr = r['xgb_roi']; pr = r['ppo_roi']; bh = r['bh_roi']
        if xr is None or pr is None:
            best = 'N/A'
        elif xr >= pr and xr >= bh:
            best = '🏆 XGBoost'
        elif pr >= xr and pr >= bh:
            best = '🏆 PPO'
        else:
            best = '🏆 B&H'
        xr_s  = f"{xr:+.2f}%" if xr is not None else 'N/A'
        pr_s  = f"{pr:+.2f}%" if pr is not None else 'N/A'
        bh_s  = f"{bh:+.2f}%"
        print(f"  {r['ticker']:<12} {r['name']:<8} {xr_s:>9} {pr_s:>9} {bh_s:>9} {best:>10}")
    print(f"{'='*72}")

    print(f"\n  📋 DETAIL: Win-rate & Trades")
    print(f"  {'─'*62}")
    print(f"  {'Ticker':<12} {'Model':<10} {'Trades':>8} {'Win Rate':>10}")
    print(f"  {'─'*62}")
    for r in all_results:
        if r['xgb_trades'] is not None:
            print(f"  {r['ticker']:<12} {'XGBoost':<10} {r['xgb_trades']:>8} {r['xgb_wr']:>9.1f}%")
        if r['ppo_trades'] is not None:
            print(f"  {r['ticker']:<12} {'PPO':<10} {r['ppo_trades']:>8} {r['ppo_wr']:>9.1f}%")
    print(f"{'='*72}")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
