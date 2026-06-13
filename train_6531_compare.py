"""
6531.TW (愛普) - 三模型比較訓練
PPO (現有) vs SAC vs XGBoost
結果顯示比較表
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC, PPO
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime
import json

TICKER = '6531.TW'
SYMBOL = '6531'
NAME   = '愛普'

# ─────────────────────────────────────────────
# Shared feature builder
# ─────────────────────────────────────────────
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
    df['price_change_1']  = df['close'].pct_change(1)
    df['price_change_5']  = df['close'].pct_change(5)
    df['price_change_10'] = df['close'].pct_change(10)
    df['volume_ma']    = df['volume'].rolling(10).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    # XGBoost target
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    features = ['sma_10','sma_30','sma_50','rsi','macd','macd_signal','macd_hist',
                'bb_pct','price_change_1','price_change_5','price_change_10',
                'volume_ratio','high_low_range']
    df = df.dropna()
    X = df[features].values
    y = df['target'].values
    return X, y, df

# ─────────────────────────────────────────────
# RL Trading Environment (shared by PPO & SAC)
# ─────────────────────────────────────────────
class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0
        self.total_trades = 0
        self.last_trade_step = 0
        self.last_total_value = self.initial_balance
        return self._obs(), {}

    def _obs(self):
        r = self.df.iloc[self.current_step]
        cp = float(r['close'])
        tv = self.balance + self.shares_held * cp
        return np.array([
            float(self.shares_held), float(self.balance), cp,
            float(r.get('sma_10', 0)), float(r.get('sma_30', 0)), float(r.get('sma_50', 0)),
            float(r.get('rsi', 50)), float(r.get('macd', 0)), float(r.get('macd_signal', 0)),
            float(r.get('bb_upper', 0)), float(r.get('bb_lower', 0)), float(r.get('volume', 0)),
            float(self.total_profit),
            (self.shares_held * cp) / tv if tv > 0 else 0,
            self.balance / tv if tv > 0 else 1,
        ], dtype=np.float32)

    def step(self, action):
        action = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        action = np.clip(action, -1.0, 1.0)
        cp = float(self.df.iloc[self.current_step]['close'])
        tv_before = self.balance + self.shares_held * cp
        traded = False

        if action > 0.05:
            s = int(int(self.balance / cp) * abs(action))
            if s > 0:
                self.balance -= s * cp; self.shares_held += s
                self.total_trades += 1; self.last_trade_step = self.current_step; traded = True
        elif action < -0.05:
            s = int(self.shares_held * abs(action))
            if s > 0:
                self.balance += s * cp; self.shares_held -= s
                self.total_trades += 1; self.last_trade_step = self.current_step; traded = True

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        tv = self.balance + self.shares_held * cp
        self.total_profit = tv - self.initial_balance

        pct = (tv - tv_before) / tv_before if tv_before > 0 else 0
        reward = pct * 10
        if traded: reward += 0.02
        idle = self.current_step - self.last_trade_step
        if idle > 30: reward -= 0.01 * (idle / 30)
        self.last_total_value = tv
        return self._obs(), reward, done, False, {}


def download_data():
    df = yf.download(TICKER, start='2015-01-01', end='2026-12-31', progress=False)
    if df.empty:
        raise ValueError("No data")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.rename(columns={'Close':'close','Volume':'volume',
                             'Open':'open','High':'high','Low':'low'}).reset_index()
    return df


def eval_rl(model, df_clean):
    """Evaluate a RL model, return (accuracy%, return%, trades)"""
    test = TradingEnv(df_clean)
    obs, _ = test.reset()
    cp_t = tp = 0
    for _ in range(len(df_clean) - 1):
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, done, _, _ = test.step(act)
        if abs(act[0]) > 0.05:
            tp += 1; cp_t += (1 if rew > 0 else 0)
        if done: break
    acc = cp_t / tp * 100 if tp > 0 else 0
    fv  = test.balance + test.shares_held * df_clean.iloc[-1]['close']
    ret = (fv - test.initial_balance) / test.initial_balance * 100
    return round(acc, 1), round(ret, 1), test.total_trades


# ─────────────────────────────────────────────
# Train SAC
# ─────────────────────────────────────────────
def train_sac(df_clean):
    print(f"\n{'='*60}\nSAC 訓練 {TICKER} ({NAME})\n{'='*60}")
    env = TradingEnv(df_clean)
    model = SAC('MlpPolicy', env,
                learning_rate=0.0003, buffer_size=100000,
                learning_starts=1000, batch_size=256,
                tau=0.005, gamma=0.99, ent_coef='auto', verbose=0)
    print("  訓練中... (200000步)")
    model.learn(total_timesteps=200000)
    fname = f'sac_{SYMBOL}_compare'
    model.save(fname)
    print(f"  保存: {fname}.zip")
    acc, ret, trades = eval_rl(model, df_clean)
    print(f"  準確度:{acc:.1f}%  回報:{ret:.1f}%  交易:{trades}")
    return acc, ret, trades


# ─────────────────────────────────────────────
# Train XGBoost
# ─────────────────────────────────────────────
def train_xgb(df_raw):
    print(f"\n{'='*60}\nXGBoost 訓練 {TICKER} ({NAME})\n{'='*60}")
    X, y, df_feat = make_features(df_raw)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"  訓練集:{len(X_train)}  測試集:{len(X_test)}")

    model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric='logloss', verbosity=0, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
    test_acc  = accuracy_score(y_test,  model.predict(X_test))  * 100
    print(f"  訓練準確度:{train_acc:.2f}%  測試準確度:{test_acc:.2f}%")

    # Simple backtest
    test_df   = df_feat.iloc[split:].copy()
    test_preds = model.predict(X_test)
    capital = 10000.0; shares = 0
    for i, pred in enumerate(test_preds):
        price = test_df.iloc[i]['close']
        if pred == 1 and capital > price:
            s = int(capital / price); capital -= s * price; shares += s
        elif pred == 0 and shares > 0:
            capital += shares * price; shares = 0
    final = capital + shares * test_df.iloc[-1]['close']
    ret = (final - 10000) / 10000 * 100
    print(f"  回測回報率:{ret:.2f}%  最終價值:${final:.2f}")

    fname = f'xgb_{SYMBOL}_compare.pkl'
    joblib.dump(model, fname)
    print(f"  保存: {fname}")
    return round(test_acc, 1), round(ret, 1)


# ─────────────────────────────────────────────
# Load existing PPO result (if available)
# ─────────────────────────────────────────────
def load_ppo_result(df_clean):
    print(f"\n{'='*60}\nPPO 評估 {TICKER} ({NAME}) (現有模型)\n{'='*60}")
    try:
        model = PPO.load(f'ppo_{SYMBOL}_improved')
        acc, ret, trades = eval_rl(model, df_clean)
        print(f"  準確度:{acc:.1f}%  回報:{ret:.1f}%  交易:{trades}")
        return acc, ret, trades
    except Exception as e:
        print(f"  ❌ 無法加載 PPO 模型: {e}")
        return None, None, None


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print(f"三模型比較訓練 — {TICKER} ({NAME})")
    print("=" * 60)

    df_raw = download_data()
    print(f"  {len(df_raw)} 天數據下載完成")

    # Prepare clean df for RL envs
    df_rl = df_raw.copy()
    df_rl['sma_10'] = df_rl['close'].rolling(10).mean()
    df_rl['sma_30'] = df_rl['close'].rolling(30).mean()
    df_rl['sma_50'] = df_rl['close'].rolling(50).mean()
    df_rl['ema_12'] = df_rl['close'].ewm(span=12).mean()
    df_rl['ema_26'] = df_rl['close'].ewm(span=26).mean()
    d = df_rl['close'].diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    df_rl['rsi'] = 100 - (100 / (1 + g / (l + 1e-10)))
    df_rl['macd'] = df_rl['ema_12'] - df_rl['ema_26']
    df_rl['macd_signal'] = df_rl['macd'].ewm(span=9).mean()
    df_rl['bb_middle'] = df_rl['close'].rolling(20).mean()
    df_rl['bb_std'] = df_rl['close'].rolling(20).std()
    df_rl['bb_upper'] = df_rl['bb_middle'] + df_rl['bb_std'] * 2
    df_rl['bb_lower'] = df_rl['bb_middle'] - df_rl['bb_std'] * 2
    df_rl = df_rl.bfill().ffill().dropna()

    # Train / evaluate
    ppo_acc, ppo_ret, ppo_trades = load_ppo_result(df_rl)
    sac_acc, sac_ret, sac_trades = train_sac(df_rl)
    xgb_acc, xgb_ret             = train_xgb(df_raw)

    # ── Comparison Table ──
    print("\n" + "=" * 60)
    print(f"{'三模型比較表':^60}")
    print("=" * 60)
    print(f"{'模型':<12}{'測試準確度':>14}{'回測回報率':>14}{'交易次數':>12}")
    print("-" * 60)

    def fmt_row(name, acc, ret, trades):
        acc_s    = f"{acc:.1f}%"    if acc    is not None else "N/A"
        ret_s    = f"{ret:+.1f}%"   if ret    is not None else "N/A"
        trades_s = str(trades)       if trades is not None else "N/A"
        print(f"{name:<12}{acc_s:>14}{ret_s:>14}{trades_s:>12}")

    fmt_row("PPO",     ppo_acc, ppo_ret, ppo_trades)
    fmt_row("SAC",     sac_acc, sac_ret, sac_trades)
    fmt_row("XGBoost", xgb_acc, xgb_ret, "N/A")
    print("=" * 60)

    # Recommend best by backtest return
    candidates = [(ppo_ret, "PPO"), (sac_ret, "SAC"), (xgb_ret, "XGBoost")]
    candidates = [(r, n) for r, n in candidates if r is not None]
    if candidates:
        best_ret, best_name = max(candidates, key=lambda x: x[0])
        print(f"\n🏆 最高回測回報: {best_name} ({best_ret:+.1f}%)")

    # Save JSON summary
    summary = {
        'ticker': TICKER, 'name': NAME,
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'PPO':     {'accuracy': ppo_acc, 'return': ppo_ret, 'trades': ppo_trades},
        'SAC':     {'accuracy': sac_acc, 'return': sac_ret, 'trades': sac_trades},
        'XGBoost': {'accuracy': xgb_acc, 'return': xgb_ret, 'trades': 'N/A'},
    }
    with open(f'compare_{SYMBOL}_3models.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n結果已保存: compare_{SYMBOL}_3models.json")
