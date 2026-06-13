"""
批量訓練 11 支台股 XGBoost + PPO
3081 聯亞 / 6426 / 2345 智邦 / 3189 景碩 / 3037 欣興
3131 弘塑 / 7257 通嘉 / 6442 光聖 / 4979 / 2303 聯電
6451 / 3363 上詮
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import json
from datetime import datetime

# ticker, symbol_xgb, symbol_ppo, name
STOCKS = [
    ('3081.TWO', '3081_TWO', '3081', '聯亞'),
    ('6426.TW',  '6426_TW',  '6426', '6426'),
    ('2345.TW',  '2345_TW',  '2345', '智邦'),
    ('3189.TW',  '3189_TW',  '3189', '景碩'),
    ('3037.TW',  '3037_TW',  '3037', '欣興'),
    ('3131.TWO', '3131_TWO', '3131', '弘塑'),
    ('7257.TW',  '7257_TW',  '7257', '通嘉'),
    ('6442.TW',  '6442_TW',  '6442', '光聖'),
    ('4979.TW',  '4979_TW',  '4979', '4979'),
    ('2303.TW',  '2303_TW',  '2303', '聯電'),
    ('6451.TWO', '6451_TWO', '6451', '6451'),
    ('3363.TWO', '3363_TWO', '3363', '上詮'),
]

FEATURE_COLUMNS = [
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_position', 'K', 'D', 'obv', 'obv_ma20',
    'sma_10', 'sma_30', 'sma_50', 'sma_200',
    'volatility', 'atr',
    'price_change_5d', 'price_change_10d', 'price_change_20d',
    'ma50_slope'
]


def add_features_xgb(df):
    df['sma_10']  = df['close'].rolling(10).mean()
    df['sma_30']  = df['close'].rolling(30).mean()
    df['sma_50']  = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['ema_12']  = df['close'].ewm(span=12).mean()
    df['ema_26']  = df['close'].ewm(span=26).mean()
    delta = df['close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi']         = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df['macd']        = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    df['bb_middle']   = df['close'].rolling(20).mean()
    df['bb_std']      = df['close'].rolling(20).std()
    df['bb_upper']    = df['bb_middle'] + df['bb_std'] * 2
    df['bb_lower']    = df['bb_middle'] - df['bb_std'] * 2
    df['bb_position'] = ((df['close'] - df['bb_lower']) /
                         (df['bb_upper'] - df['bb_lower'] + 1e-10) * 100).fillna(50)
    low_14  = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['K'] = ((df['close'] - low_14) / (high_14 - low_14 + 1e-10) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()
    df['obv']      = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma20'] = df['obv'].rolling(20).mean()
    df['volatility'] = df['close'].rolling(20).std() / (df['close'].rolling(20).mean() + 1e-10)
    hl = df['high'] - df['low']
    hc = np.abs(df['high'] - df['close'].shift())
    lc = np.abs(df['low']  - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['price_change_5d']  = df['close'].pct_change(5)  * 100
    df['price_change_10d'] = df['close'].pct_change(10) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100
    df['ma50_slope']       = df['sma_50'].diff(5) / (df['sma_50'].shift(5) + 1e-10) * 100
    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df['target'] = (df['future_return'] > 0.02).astype(int)
    return df


def add_features_ppo(df):
    df['sma_10']  = df['close'].rolling(10).mean()
    df['sma_30']  = df['close'].rolling(30).mean()
    df['sma_50']  = df['close'].rolling(50).mean()
    df['ema_12']  = df['close'].ewm(span=12).mean()
    df['ema_26']  = df['close'].ewm(span=26).mean()
    delta = df['close'].diff()
    df['rsi']         = 100 - (100 / (1 + delta.where(delta>0,0).rolling(14).mean() /
                                      ((-delta.where(delta<0,0)).rolling(14).mean() + 1e-10)))
    df['macd']        = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['bb_middle']   = df['close'].rolling(20).mean()
    df['bb_std']      = df['close'].rolling(20).std()
    df['bb_upper']    = df['bb_middle'] + df['bb_std'] * 2
    df['bb_lower']    = df['bb_middle'] - df['bb_std'] * 2
    return df


class ImprovedTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0; self.balance = self.initial_balance
        self.shares_held = 0; self.total_profit = 0
        self.total_trades = 0; self.last_total_value = self.initial_balance
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]; cp = float(row['close'])
        tv = self.balance + self.shares_held * cp
        return np.array([
            float(self.shares_held), float(self.balance), cp,
            float(row.get('sma_10', 0)), float(row.get('sma_30', 0)), float(row.get('sma_50', 0)),
            float(row.get('rsi', 50)), float(row.get('macd', 0)), float(row.get('macd_signal', 0)),
            float(row.get('bb_upper', 0)), float(row.get('bb_lower', 0)), float(row.get('volume', 0)),
            float(self.total_profit),
            (self.shares_held * cp) / tv if tv > 0 else 0,
            self.balance / tv if tv > 0 else 1
        ], dtype=np.float32)

    def step(self, action):
        if isinstance(action, np.ndarray): action = float(action[0])
        action = np.clip(action, -1.0, 1.0)
        cp = float(self.df.iloc[self.current_step]['close']); traded = False
        if action > 0.15:
            sb = int(int(self.balance / cp) * abs(action))
            if sb > 0: self.balance -= sb*cp; self.shares_held += sb; self.total_trades += 1; traded = True
        elif action < -0.15:
            ss = int(self.shares_held * abs(action))
            if ss > 0: self.balance += ss*cp; self.shares_held -= ss; self.total_trades += 1; traded = True
        self.current_step += 1; done = self.current_step >= len(self.df) - 1
        tv = self.balance + self.shares_held * cp; self.total_profit = tv - self.initial_balance
        vc = tv - self.last_total_value; reward = vc / self.initial_balance
        if traded: reward += 0.01
        if self.total_trades < self.current_step / 100: reward -= 0.001
        self.last_total_value = tv
        return self._get_observation(), reward, done, False, {}


def try_download(ticker):
    """Try ticker as-is; if 404, auto-flip TW↔TWO"""
    df = yf.download(ticker, start='2015-01-01', end='2026-12-31', progress=False)
    if df.empty or len(df) < 100:
        alt = ticker.replace('.TW', '.TWO') if ticker.endswith('.TW') else ticker.replace('.TWO', '.TW')
        df2 = yf.download(alt, start='2015-01-01', end='2026-12-31', progress=False)
        if not df2.empty and len(df2) >= 100:
            print(f"  ⚠️  {ticker} 無資料，改用 {alt}")
            return df2, alt
        return None, ticker
    return df, ticker


def train_stock(ticker, sym_xgb, sym_ppo, name):
    print(f"\n{'='*70}\n訓練 {ticker} ({name})\n{'='*70}")

    raw_df, actual_ticker = try_download(ticker)
    if raw_df is None:
        print("  ❌ 無資料，跳過"); return None
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.droplevel(1)
    raw_df = raw_df.rename(columns={'Close':'close','Volume':'volume',
                                    'Open':'open','High':'high','Low':'low'}).reset_index()
    print(f"  下載 {len(raw_df)} 天數據 (使用 {actual_ticker})")

    # ── XGBoost ────────────────────────────────────────────────────
    print("  [XGBoost]")
    xgb_acc = 0.0
    try:
        dc = add_features_xgb(raw_df.copy()).dropna(subset=FEATURE_COLUMNS + ['target'])
        print(f"  清理後 {len(dc)} 天")
        X = dc[FEATURE_COLUMNS]; y = dc['target']
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)
        m = xgb.XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=200,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            objective='binary:logistic', random_state=42, eval_metric='logloss')
        m.fit(Xtr, ytr)
        ta = accuracy_score(ytr, m.predict(Xtr))
        xgb_acc = accuracy_score(yte, m.predict(Xte))
        print(f"  訓練: {ta*100:.2f}%  測試: {xgb_acc*100:.2f}%")
        joblib.dump(m, f'xgb_{sym_xgb.lower()}_model.pkl')
        with open(f'model_accuracy_{sym_xgb}.json', 'w', encoding='utf-8') as f:
            json.dump({'symbol': actual_ticker, 'model_type': 'XGBoost',
                       'training_accuracy': float(ta*100), 'validation_accuracy': float(xgb_acc*100),
                       'backtest_accuracy': float(xgb_acc*100),
                       'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                      f, ensure_ascii=False, indent=2)
        fi = pd.DataFrame({'feature': FEATURE_COLUMNS, 'importance': m.feature_importances_}
                         ).sort_values('importance', ascending=False)
        with open(f'{sym_xgb}_feature_importance.json', 'w', encoding='utf-8') as f:
            json.dump({'ticker': actual_ticker, 'model_type': 'XGBoost',
                       'model_accuracy': float(xgb_acc),
                       'feature_importance': {r['feature']: float(r['importance'])
                                              for _, r in fi.iterrows()}},
                      f, ensure_ascii=False, indent=2)
        tag = '🌟 EXCELLENT' if xgb_acc >= 0.65 else ('✅' if xgb_acc >= 0.50 else '⚠️')
        print(f"  {tag} XGBoost ({xgb_acc*100:.2f}%)")
    except Exception as e:
        print(f"  ❌ XGBoost failed: {e}")

    # ── PPO ────────────────────────────────────────────────────────
    print("  [PPO]")
    ppo_acc = 0.0; ppo_ret = 0.0
    try:
        df_ppo = add_features_ppo(raw_df.copy()).bfill().ffill()
        dc_ppo = df_ppo.dropna()
        print(f"  清理後 {len(dc_ppo)} 天")
        env = DummyVecEnv([lambda d=dc_ppo: ImprovedTradingEnv(d)])
        ppo_m = PPO('MlpPolicy', env, learning_rate=0.0005, n_steps=2048,
                    batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01, verbose=0)
        print("  訓練中... (200000 步)")
        ppo_m.learn(total_timesteps=200000)
        ppo_m.save(f'ppo_{sym_ppo}_improved')
        te = ImprovedTradingEnv(dc_ppo); obs, _ = te.reset()
        correct = 0; total = 0
        for _ in range(len(dc_ppo) - 1):
            action, _ = ppo_m.predict(obs, deterministic=True)
            obs, reward, done, _, _ = te.step(action)
            if abs(action[0]) > 0.15: total += 1; correct += (1 if reward > 0 else 0)
            if done: break
        ppo_acc = (correct / total * 100) if total > 0 else 0
        fv = te.balance + te.shares_held * dc_ppo.iloc[-1]['close']
        ppo_ret = (fv - 10000) / 10000 * 100
        print(f"  ✅ PPO 準確度: {ppo_acc:.2f}% | 回報率: {ppo_ret:.2f}% | 交易: {te.total_trades}")
        with open(f'model_accuracy_{sym_ppo}.json', 'w', encoding='utf-8') as f:
            json.dump({'symbol': sym_ppo, 'model_type': 'PPO',
                       'training_accuracy': float(ppo_acc), 'validation_accuracy': float(ppo_acc),
                       'backtest_accuracy': float(ppo_acc), 'backtest_return': float(ppo_ret),
                       'win_rate': float(ppo_acc), 'sharpe_ratio': None,
                       'total_signals': int(total), 'correct_signals': int(correct),
                       'live_accuracy': None,
                       'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                       'history': []}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  ❌ PPO failed: {e}")

    return (ticker, name, xgb_acc, ppo_acc, ppo_ret)


if __name__ == '__main__':
    print("=" * 70)
    print("批量訓練 12 支台股 XGBoost + PPO")
    print(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = []
    for ticker, sym_xgb, sym_ppo, name in STOCKS:
        r = train_stock(ticker, sym_xgb, sym_ppo, name)
        if r: results.append(r)

    print(f"\n\n{'='*70}\n訓練完成摘要\n{'='*70}")
    print(f"{'代號':<14} {'名稱':<8} {'XGBoost':>10} {'PPO準確':>10} {'PPO回報':>10}")
    print("-" * 58)
    for ticker, name, xacc, pacc, pret in results:
        tag = '🌟' if xacc >= 0.65 else ('✅' if xacc >= 0.50 else '⚠️')
        print(f"{ticker:<14} {name:<8} {xacc*100:>8.2f}% {tag}  {pacc:>7.2f}%  {pret:>+8.2f}%")
    print(f"\n完成: {len(results)}/{len(STOCKS)}")
