"""
ROI Backtest: XGBoost vs PPO — Batch 3 (14 TW Stocks)
2492 華新科, 3026 禾伸堂, 2472 立隆電, 2375 凱美, 2327 國巨,
6449 鉅邦,   6175 立敦,   8042 金山電, 6127 九豪, 3357 臺慶科,
2478 大毅,   3090 日電貿, 6224 聚鼎,   8043 蜜望寶
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
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# ─── Stock list (exchange confirmed from training) ────────────────────────────
STOCKS = [
    {'code': '2492', 'ticker': '2492.TW',  'name': '華新科'},
    {'code': '3026', 'ticker': '3026.TW',  'name': '禾伸堂'},
    {'code': '2472', 'ticker': '2472.TW',  'name': '立隆電'},
    {'code': '2375', 'ticker': '2375.TW',  'name': '凱美'},
    {'code': '2327', 'ticker': '2327.TW',  'name': '國巨'},
    {'code': '6449', 'ticker': '6449.TW',  'name': '鉅邦'},
    {'code': '6175', 'ticker': '6175.TWO', 'name': '立敦'},
    {'code': '8042', 'ticker': '8042.TWO', 'name': '金山電'},
    {'code': '6127', 'ticker': '6127.TWO', 'name': '九豪'},
    {'code': '3357', 'ticker': '3357.TWO', 'name': '臺慶科'},
    {'code': '2478', 'ticker': '2478.TW',  'name': '大毅'},
    {'code': '3090', 'ticker': '3090.TW',  'name': '日電貿'},
    {'code': '6224', 'ticker': '6224.TW',  'name': '聚鼎'},
    {'code': '8043', 'ticker': '8043.TWO', 'name': '蜜望寶'},
]

INITIAL_BALANCE = 100_000   # NT$100,000
END_DATE = '2026-05-29'

FEATURE_COLUMNS = [
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_position', 'K', 'D', 'obv', 'obv_ma20',
    'sma_10', 'sma_30', 'sma_50', 'sma_200',
    'volatility', 'atr',
    'price_change_5d', 'price_change_10d', 'price_change_20d',
    'ma50_slope'
]


# ─── Technical indicators ─────────────────────────────────────────────────────
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
    df['future_return']    = df['close'].shift(-5) / df['close'] - 1
    df['target']           = (df['future_return'] > 0.02).astype(int)
    return df.bfill().ffill()


# ─── XGBoost ROI backtest ─────────────────────────────────────────────────────
def backtest_xgboost(model, test_df, initial_balance=INITIAL_BALANCE):
    balance   = initial_balance
    shares    = 0
    trades    = 0
    wins      = 0
    buy_price = None
    equity_curve = [balance]

    X     = test_df[FEATURE_COLUMNS]
    preds = model.predict(X)

    for i, (_, row) in enumerate(test_df.iterrows()):
        price = float(row['close'])
        sig   = int(preds[i])

        if sig == 1 and balance > price:
            shares_to_buy = int(balance // price)
            if shares_to_buy > 0:
                balance   -= shares_to_buy * price
                shares    += shares_to_buy
                buy_price  = price
                trades    += 1
        elif sig == 0 and shares > 0:
            balance += shares * price
            if buy_price and price > buy_price:
                wins += 1
            shares    = 0
            buy_price = None

        equity_curve.append(balance + shares * price)

    if shares > 0:
        balance += shares * float(test_df.iloc[-1]['close'])

    roi      = (balance - initial_balance) / initial_balance * 100
    win_rate = wins / trades * 100 if trades > 0 else 0.0
    bh_roi   = (float(test_df.iloc[-1]['close']) /
                float(test_df.iloc[0]['close']) - 1) * 100
    return roi, trades, win_rate, bh_roi, equity_curve


# ─── PPO TradingEnv ───────────────────────────────────────────────────────────
class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=INITIAL_BALANCE):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.balance  = float(self.initial_balance)
        self.shares   = 0
        self.profit   = 0.0
        return self._obs(), {}

    def _obs(self):
        r  = self.df.iloc[self.step_idx]
        p  = float(r['close'])
        tv = self.balance + self.shares * p
        return np.array([
            float(self.shares), float(self.balance), p,
            float(r.get('sma_10', 0)),   float(r.get('sma_30', 0)),
            float(r.get('sma_50', 0)),   float(r.get('rsi', 50)),
            float(r.get('macd', 0)),     float(r.get('macd_signal', 0)),
            float(r.get('bb_upper', 0)), float(r.get('bb_lower', 0)),
            float(r.get('volume', 0)),   float(self.profit),
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
        self.profit   = (self.balance + self.shares * p) - self.initial_balance
        self.step_idx += 1
        done   = self.step_idx >= len(self.df) - 1
        reward = self.profit / self.initial_balance
        if abs(a) > 0.1: reward += 0.01
        if self.balance > (self.balance + self.shares * p) * 0.9: reward -= 0.005
        return self._obs(), reward, done, False, {}


def backtest_ppo(model, test_df, initial_balance=INITIAL_BALANCE):
    env = TradingEnv(test_df.reset_index(drop=True), initial_balance=initial_balance)
    obs, _ = env.reset()
    trades  = 0
    wins    = 0
    buy_price   = None
    last_shares = 0
    equity_curve = [initial_balance]

    for _ in range(len(test_df) - 1):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(act)
        a     = float(act[0])
        price = float(env.df.iloc[env.step_idx - 1]['close'])

        if a > 0.1 and env.shares > last_shares:
            buy_price = price
            trades += 1
        elif a < -0.1 and env.shares < last_shares and last_shares > 0:
            if buy_price and price > buy_price:
                wins += 1
            buy_price = None

        last_shares = env.shares
        equity_curve.append(env.balance + env.shares * price)
        if done:
            break

    final    = env.balance + env.shares * float(test_df.iloc[-1]['close'])
    roi      = (final - initial_balance) / initial_balance * 100
    win_rate = wins / trades * 100 if trades > 0 else 0.0
    bh_roi   = (float(test_df.iloc[-1]['close']) /
                float(test_df.iloc[0]['close']) - 1) * 100
    return roi, trades, win_rate, bh_roi, equity_curve


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 76)
    print("  💰 ROI BACKTEST: XGBoost vs PPO — Batch 3 (14 TW Stocks)")
    print(f"  Initial capital: NT${INITIAL_BALANCE:,}  |  Test set = last 20%  |  End: {END_DATE}")
    print("=" * 76)

    all_results = []

    for s in STOCKS:
        code, ticker, name = s['code'], s['ticker'], s['name']
        suffix = 'two' if '.TWO' in ticker else 'tw'

        print(f"\n{'─'*76}")
        print(f"  {ticker} ({name})")
        print(f"{'─'*76}")

        # Download
        df = yf.download(ticker, start='2015-01-01', end=END_DATE, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty:
            print(f"  ❌ No data, skipping.")
            continue
        df = df.rename(columns={'Close':'close','Volume':'volume',
                                'Open':'open','High':'high','Low':'low'}).reset_index()

        df = add_indicators(df)
        df_clean = df.dropna(subset=FEATURE_COLUMNS + ['target'])

        split      = int(len(df_clean) * 0.8)
        test_df    = df_clean.iloc[split:].copy().reset_index(drop=True)
        test_start = str(test_df.iloc[0]['Date'])[:10]
        test_end   = str(test_df.iloc[-1]['Date'])[:10]
        print(f"  Data: {len(df_clean)} days  |  Test: {len(test_df)} days  [{test_start} → {test_end}]")

        bh_roi = None

        # XGBoost backtest
        xgb_file = f'xgb_{code}_{suffix}_model.pkl'
        xgb_roi = xgb_trades = xgb_wr = None
        try:
            xgb_model = joblib.load(xgb_file)
            xgb_roi, xgb_trades, xgb_wr, bh_roi, _ = backtest_xgboost(xgb_model, test_df)
            print(f"\n  [XGBoost]  ROI: {xgb_roi:+.2f}%  |  Trades: {xgb_trades}  |  Win: {xgb_wr:.1f}%  |  Final: NT${INITIAL_BALANCE*(1+xgb_roi/100):,.0f}")
        except FileNotFoundError:
            print(f"  ⚠  XGBoost model not found: {xgb_file}")
        except Exception as e:
            print(f"  ❌ XGBoost error: {e}")

        # PPO backtest
        ppo_file = f'ppo_{code}_{suffix}_improved'
        ppo_roi = ppo_trades = ppo_wr = None
        try:
            ppo_model = PPO.load(ppo_file)
            ppo_roi, ppo_trades, ppo_wr, bh_roi_p, _ = backtest_ppo(ppo_model, test_df)
            if bh_roi is None: bh_roi = bh_roi_p
            print(f"  [PPO]      ROI: {ppo_roi:+.2f}%  |  Trades: {ppo_trades}  |  Win: {ppo_wr:.1f}%  |  Final: NT${INITIAL_BALANCE*(1+ppo_roi/100):,.0f}")
        except FileNotFoundError:
            print(f"  ⚠  PPO model not found: {ppo_file}")
        except Exception as e:
            print(f"  ❌ PPO error: {e}")

        # Buy & Hold
        if bh_roi is None:
            bh_roi = (float(test_df.iloc[-1]['close']) / float(test_df.iloc[0]['close']) - 1) * 100
        print(f"  [Buy&Hold] ROI: {bh_roi:+.2f}%  |  Final: NT${INITIAL_BALANCE*(1+bh_roi/100):,.0f}")

        all_results.append({
            'ticker': ticker, 'name': name,
            'xgb_roi': xgb_roi, 'xgb_trades': xgb_trades, 'xgb_wr': xgb_wr,
            'ppo_roi': ppo_roi, 'ppo_trades': ppo_trades, 'ppo_wr': ppo_wr,
            'bh_roi':  bh_roi,
            'test_start': test_start, 'test_end': test_end,
        })

    # ─── Summary tables ──────────────────────────────────────────────────────
    print(f"\n{'='*76}")
    print("  📊 ROI COMPARISON TABLE  (Initial capital: NT$100,000)")
    print(f"{'='*76}")
    print(f"  {'#':>3}  {'Ticker':<11} {'Name':<9} {'XGB ROI':>9} {'PPO ROI':>9} {'B&H ROI':>9} {'Best':>12}")
    print(f"  {'─'*70}")

    xgb_wins = ppo_wins = bh_wins = 0
    xgb_rois = []; ppo_rois = []; bh_rois = []

    for i, r in enumerate(all_results, 1):
        xr = r['xgb_roi']; pr = r['ppo_roi']; bh = r['bh_roi']
        candidates = {}
        if xr is not None: candidates['XGBoost'] = xr
        if pr is not None: candidates['PPO']      = pr
        candidates['B&H'] = bh
        best_name = max(candidates, key=candidates.get)
        if best_name == 'XGBoost': xgb_wins += 1
        elif best_name == 'PPO':   ppo_wins  += 1
        else:                       bh_wins   += 1
        if xr is not None: xgb_rois.append(xr)
        if pr is not None: ppo_rois.append(pr)
        bh_rois.append(bh)
        xr_s = f"{xr:+.2f}%" if xr is not None else '  N/A '
        pr_s = f"{pr:+.2f}%" if pr is not None else '  N/A '
        print(f"  {i:>3}  {r['ticker']:<11} {r['name']:<9} {xr_s:>9} {pr_s:>9} {bh:>+9.2f}% {'🏆 '+best_name:>12}")

    print(f"  {'─'*70}")
    avg_xgb = f"{sum(xgb_rois)/len(xgb_rois):+.2f}%" if xgb_rois else 'N/A'
    avg_ppo = f"{sum(ppo_rois)/len(ppo_rois):+.2f}%" if ppo_rois else 'N/A'
    avg_bh  = f"{sum(bh_rois)/len(bh_rois):+.2f}%"
    print(f"  {'':>3}  {'Average':<11} {'':<9} {avg_xgb:>9} {avg_ppo:>9} {avg_bh:>9}")
    print(f"  {'':>3}  {'Wins':<11} {'':<9} {xgb_wins:>9} {ppo_wins:>9} {bh_wins:>9}")
    print(f"{'='*76}")

    print(f"\n  📋 DETAIL: Win-rate & Trades")
    print(f"  {'─'*64}")
    print(f"  {'Ticker':<11} {'Name':<9} {'Model':<10} {'Trades':>8} {'WinRate':>9} {'ROI':>9}")
    print(f"  {'─'*64}")
    for r in all_results:
        if r['xgb_roi'] is not None:
            print(f"  {r['ticker']:<11} {r['name']:<9} {'XGBoost':<10} {r['xgb_trades']:>8} {r['xgb_wr']:>8.1f}% {r['xgb_roi']:>+8.2f}%")
        if r['ppo_roi'] is not None:
            print(f"  {r['ticker']:<11} {r['name']:<9} {'PPO':<10} {r['ppo_trades']:>8} {r['ppo_wr']:>8.1f}% {r['ppo_roi']:>+8.2f}%")
        print(f"  {r['ticker']:<11} {r['name']:<9} {'Buy&Hold':<10} {'—':>8} {'—':>9} {r['bh_roi']:>+8.2f}%")
    print(f"{'='*76}")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
