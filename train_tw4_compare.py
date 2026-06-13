"""
四支台股 三模型比較訓練
2313 (華通) / 2344 (華邦電) / 6770 (力積電) / 2308 (台達電)
PPO (現有模型) vs SAC vs XGBoost
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

STOCKS = [
    ('2313.TW', '2313_tw', '華通電腦'),
    ('2344.TW', '2344_tw', '華邦電子'),
    ('6770.TW', '6770_tw', '力積電'),
    ('2308.TW', '2308_tw', '台達電'),
]

# ─────────────────────────────────────────────
# Feature builder (XGBoost)
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
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    features = ['sma_10','sma_30','sma_50','rsi','macd','macd_signal','macd_hist',
                'bb_pct','price_change_1','price_change_5','price_change_10',
                'volume_ratio','high_low_range']
    df = df.dropna()
    return df[features].values, df['target'].values, df


# ─────────────────────────────────────────────
# RL Environment (shared by PPO & SAC)
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


def download_data(ticker):
    df = yf.download(ticker, start='2015-01-01', end='2026-12-31', progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.rename(columns={'Close':'close','Volume':'volume',
                             'Open':'open','High':'high','Low':'low'}).reset_index()
    return df


def prep_rl_df(df_raw):
    df = df_raw.copy()
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
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + df['bb_std'] * 2
    df['bb_lower'] = df['bb_middle'] - df['bb_std'] * 2
    return df.bfill().ffill().dropna()


def eval_rl(model, df_clean):
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
# Per-stock training
# ─────────────────────────────────────────────
def run_stock(ticker, symbol, name):
    print(f"\n{'='*65}")
    print(f"  {ticker} ({name})")
    print(f"{'='*65}")

    result = {'ticker': ticker, 'name': name}

    try:
        df_raw = download_data(ticker)
        print(f"  {len(df_raw)} 天數據")
        df_clean = prep_rl_df(df_raw)
        print(f"  清理後: {len(df_clean)} 天")
    except Exception as e:
        print(f"  ❌ 下載失敗: {e}")
        return result

    # ── PPO (existing model) ──
    print(f"\n  [PPO] 加載現有模型...")
    try:
        ppo_path = f'ppo_{symbol}_improved'
        ppo_model = PPO.load(ppo_path)
        ppo_acc, ppo_ret, ppo_trades = eval_rl(ppo_model, df_clean)
        print(f"  [PPO] 準確度:{ppo_acc:.1f}%  回報:{ppo_ret:.1f}%  交易:{ppo_trades}")
        result['PPO'] = {'accuracy': ppo_acc, 'return': ppo_ret, 'trades': ppo_trades}
    except Exception as e:
        print(f"  [PPO] ❌ {e}")
        result['PPO'] = None

    # ── SAC ──
    print(f"\n  [SAC] 訓練中... (200000步)")
    try:
        env = TradingEnv(df_clean)
        sac_model = SAC('MlpPolicy', env,
                        learning_rate=0.0003, buffer_size=100000,
                        learning_starts=1000, batch_size=256,
                        tau=0.005, gamma=0.99, ent_coef='auto', verbose=0)
        sac_model.learn(total_timesteps=200000)
        sac_fname = f'sac_{symbol}_compare'
        sac_model.save(sac_fname)
        sac_acc, sac_ret, sac_trades = eval_rl(sac_model, df_clean)
        print(f"  [SAC] 準確度:{sac_acc:.1f}%  回報:{sac_ret:.1f}%  交易:{sac_trades}  保存:{sac_fname}.zip")
        result['SAC'] = {'accuracy': sac_acc, 'return': sac_ret, 'trades': sac_trades}
    except Exception as e:
        print(f"  [SAC] ❌ {e}")
        result['SAC'] = None

    # ── XGBoost ──
    print(f"\n  [XGBoost] 訓練中...")
    try:
        X, y, df_feat = make_features(df_raw)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        xgb_model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   eval_metric='logloss', verbosity=0, random_state=42)
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        train_acc = accuracy_score(y_train, xgb_model.predict(X_train)) * 100
        test_acc  = accuracy_score(y_test,  xgb_model.predict(X_test))  * 100

        # Backtest
        test_df    = df_feat.iloc[split:].copy()
        test_preds = xgb_model.predict(X_test)
        capital = 10000.0; shares = 0
        for i, pred in enumerate(test_preds):
            price = test_df.iloc[i]['close']
            if pred == 1 and capital > price:
                s = int(capital / price); capital -= s * price; shares += s
            elif pred == 0 and shares > 0:
                capital += shares * price; shares = 0
        final = capital + shares * test_df.iloc[-1]['close']
        xgb_ret = (final - 10000) / 10000 * 100

        xgb_fname = f'xgb_{symbol}_compare.pkl'
        joblib.dump(xgb_model, xgb_fname)
        print(f"  [XGBoost] 訓練:{train_acc:.1f}%  測試:{test_acc:.1f}%  回報:{xgb_ret:.1f}%  保存:{xgb_fname}")
        result['XGBoost'] = {'train_accuracy': round(train_acc, 1),
                              'test_accuracy': round(test_acc, 1),
                              'return': round(xgb_ret, 1)}
    except Exception as e:
        print(f"  [XGBoost] ❌ {e}")
        result['XGBoost'] = None

    return result


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("四支台股 — PPO vs SAC vs XGBoost 比較訓練")
    print("=" * 65)

    all_results = []
    for ticker, symbol, name in STOCKS:
        r = run_stock(ticker, symbol, name)
        all_results.append(r)

    # ── Comparison Table ──
    print("\n\n" + "=" * 65)
    print(f"{'最終比較表':^65}")
    print("=" * 65)
    print(f"{'股票':<14}{'模型':<10}{'準確度':>10}{'回測回報':>12}{'交易次數':>10}")
    print("-" * 65)

    best_per_stock = {}
    for r in all_results:
        label = f"{r['ticker']} {r['name']}"
        models = []

        ppo = r.get('PPO')
        sac = r.get('SAC')
        xgb = r.get('XGBoost')

        def row(name, acc, ret, trades='-'):
            acc_s    = f"{acc:.1f}%"   if acc   is not None else "N/A"
            ret_s    = f"{ret:+.1f}%"  if ret   is not None else "N/A"
            trades_s = str(trades)
            print(f"{label:<14}{name:<10}{acc_s:>10}{ret_s:>12}{trades_s:>10}")
            label_blank = ""  # subsequent rows blank label
            return ret

        if ppo:
            r1 = row("PPO", ppo['accuracy'], ppo['return'], ppo['trades'])
            models.append(('PPO', ppo['return']))
            label = ""
        if sac:
            row("SAC", sac['accuracy'], sac['return'], sac['trades'])
            models.append(('SAC', sac['return']))
            label = ""
        if xgb:
            row("XGBoost", xgb['test_accuracy'], xgb['return'])
            models.append(('XGBoost', xgb['return']))
            label = ""

        if models:
            best_name, best_ret = max(models, key=lambda x: x[1])
            best_per_stock[r['ticker']] = (best_name, best_ret)
        print("-" * 65)

    print("\n🏆 各股最佳模型 (依回測回報):")
    for ticker, (model, ret) in best_per_stock.items():
        print(f"   {ticker}: {model}  ({ret:+.1f}%)")

    # Save JSON
    out_file = f'compare_tw4_3models_{datetime.now().strftime("%Y%m%d%H%M")}.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n結果已保存: {out_file}")
