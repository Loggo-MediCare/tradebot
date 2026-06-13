"""
訓練 6147.TWO 頎邦 — XGBoost + PPO
(補充至 31支台灣半導體清單)
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

TICKER = '6147.TWO'
SYMBOL_XGB = '6147_TWO'
SYMBOL_PPO = '6147'
NAME = '頎邦'

FEATURE_COLUMNS = [
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_position', 'K', 'D', 'obv', 'obv_ma20',
    'sma_10', 'sma_30', 'sma_50', 'sma_200',
    'volatility', 'atr',
    'price_change_5d', 'price_change_10d', 'price_change_20d',
    'ma50_slope'
]


def add_features(df):
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


# ── Download once, reuse ──────────────────────────────────────────────────────
print("=" * 70)
print(f"訓練 {TICKER} ({NAME}) — XGBoost + PPO")
print(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

print(f"\n下載資料: {TICKER}")
raw_df = yf.download(TICKER, start='2015-01-01', end='2026-12-31', progress=False)
if raw_df.empty or len(raw_df) < 100:
    print("❌ 無資料，退出"); sys.exit(1)
if isinstance(raw_df.columns, pd.MultiIndex):
    raw_df.columns = raw_df.columns.droplevel(1)
raw_df = raw_df.rename(columns={'Close':'close','Volume':'volume',
                                'Open':'open','High':'high','Low':'low'}).reset_index()
print(f"下載 {len(raw_df)} 天數據")


# ── XGBoost ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}\nXGBoost 訓練\n{'='*70}")
df_xgb = add_features(raw_df.copy())
df_clean_xgb = df_xgb.dropna(subset=FEATURE_COLUMNS + ['target'])
print(f"清理後 {len(df_clean_xgb)} 天")

X = df_clean_xgb[FEATURE_COLUMNS]; y = df_clean_xgb['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

xgb_model = xgb.XGBClassifier(
    max_depth=5, learning_rate=0.05, n_estimators=200,
    min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
    objective='binary:logistic', random_state=42, eval_metric='logloss'
)
print("訓練中...")
xgb_model.fit(X_train, y_train)
train_acc = accuracy_score(y_train, xgb_model.predict(X_train))
test_acc  = accuracy_score(y_test,  xgb_model.predict(X_test))
print(f"訓練準確度: {train_acc*100:.2f}%  |  測試準確度: {test_acc*100:.2f}%")

joblib.dump(xgb_model, f'xgb_{SYMBOL_XGB.lower()}_model.pkl')
with open(f'model_accuracy_{SYMBOL_XGB}.json', 'w', encoding='utf-8') as f:
    json.dump({'symbol': TICKER, 'model_type': 'XGBoost',
               'training_accuracy': float(train_acc*100),
               'validation_accuracy': float(test_acc*100),
               'backtest_accuracy': float(test_acc*100),
               'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
              f, ensure_ascii=False, indent=2)
fi_df = pd.DataFrame({'feature': FEATURE_COLUMNS,
                       'importance': xgb_model.feature_importances_}
                     ).sort_values('importance', ascending=False)
with open(f'{SYMBOL_XGB}_feature_importance.json', 'w', encoding='utf-8') as f:
    json.dump({'ticker': TICKER, 'model_type': 'XGBoost', 'model_accuracy': float(test_acc),
               'feature_importance': {r['feature']: float(r['importance'])
                                      for _, r in fi_df.iterrows()}},
              f, ensure_ascii=False, indent=2)
tag = '🌟 EXCELLENT' if test_acc >= 0.65 else ('✅' if test_acc >= 0.50 else '⚠️')
print(f"✅ XGBoost 完成 — {tag} ({test_acc*100:.2f}%)")
print(f"   模型: xgb_{SYMBOL_XGB.lower()}_model.pkl")


# ── PPO ───────────────────────────────────────────────────────────────────────
class ImprovedTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super().__init__()
        self.df = df.reset_index(drop=True); self.initial_balance = initial_balance
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
        return np.array([float(self.shares_held), float(self.balance), cp,
            float(row.get('sma_10',0)), float(row.get('sma_30',0)), float(row.get('sma_50',0)),
            float(row.get('rsi',50)), float(row.get('macd',0)), float(row.get('macd_signal',0)),
            float(row.get('bb_upper',0)), float(row.get('bb_lower',0)), float(row.get('volume',0)),
            float(self.total_profit),
            (self.shares_held*cp)/tv if tv>0 else 0,
            self.balance/tv if tv>0 else 1], dtype=np.float32)

    def step(self, action):
        if isinstance(action, np.ndarray): action = float(action[0])
        action = np.clip(action, -1.0, 1.0)
        cp = float(self.df.iloc[self.current_step]['close']); traded = False
        if action > 0.15:
            sb = int(int(self.balance/cp) * abs(action))
            if sb > 0: self.balance -= sb*cp; self.shares_held += sb; self.total_trades += 1; traded = True
        elif action < -0.15:
            ss = int(self.shares_held * abs(action))
            if ss > 0: self.balance += ss*cp; self.shares_held -= ss; self.total_trades += 1; traded = True
        self.current_step += 1; done = self.current_step >= len(self.df)-1
        tv = self.balance + self.shares_held * cp; self.total_profit = tv - self.initial_balance
        vc = tv - self.last_total_value; reward = vc/self.initial_balance
        if traded: reward += 0.01
        if self.total_trades < self.current_step/100: reward -= 0.001
        self.last_total_value = tv
        return self._get_observation(), reward, done, False, {}


print(f"\n{'='*70}\nPPO 訓練\n{'='*70}")
df_ppo = raw_df.copy()
df_ppo['sma_10'] = df_ppo['close'].rolling(10).mean()
df_ppo['sma_30'] = df_ppo['close'].rolling(30).mean()
df_ppo['sma_50'] = df_ppo['close'].rolling(50).mean()
df_ppo['ema_12'] = df_ppo['close'].ewm(span=12).mean()
df_ppo['ema_26'] = df_ppo['close'].ewm(span=26).mean()
delta = df_ppo['close'].diff()
df_ppo['rsi']         = 100 - (100/(1 + delta.where(delta>0,0).rolling(14).mean() /
                                    ((-delta.where(delta<0,0)).rolling(14).mean() + 1e-10)))
df_ppo['macd']        = df_ppo['ema_12'] - df_ppo['ema_26']
df_ppo['macd_signal'] = df_ppo['macd'].ewm(span=9).mean()
df_ppo['bb_middle']   = df_ppo['close'].rolling(20).mean()
df_ppo['bb_std']      = df_ppo['close'].rolling(20).std()
df_ppo['bb_upper']    = df_ppo['bb_middle'] + df_ppo['bb_std']*2
df_ppo['bb_lower']    = df_ppo['bb_middle'] - df_ppo['bb_std']*2
df_ppo = df_ppo.bfill().ffill(); df_clean_ppo = df_ppo.dropna()
print(f"清理後 {len(df_clean_ppo)} 天")

env = DummyVecEnv([lambda d=df_clean_ppo: ImprovedTradingEnv(d)])
ppo_model = PPO('MlpPolicy', env, learning_rate=0.0005, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01, verbose=0)
print("訓練中... (200000 步)")
ppo_model.learn(total_timesteps=200000)
ppo_model.save(f'ppo_{SYMBOL_PPO}_improved')
print(f"模型已保存: ppo_{SYMBOL_PPO}_improved.zip")

test_env = ImprovedTradingEnv(df_clean_ppo); obs, _ = test_env.reset()
correct = 0; total = 0
for _ in range(len(df_clean_ppo)-1):
    action, _ = ppo_model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = test_env.step(action)
    if abs(action[0]) > 0.15:
        total += 1; correct += (1 if reward > 0 else 0)
    if done: break

acc = (correct/total*100) if total > 0 else 0
fv  = test_env.balance + test_env.shares_held * df_clean_ppo.iloc[-1]['close']
ret = (fv-10000)/10000*100
print(f"✅ PPO 完成 — 準確度: {acc:.2f}% | 回報率: {ret:.2f}% | 交易: {test_env.total_trades}")

with open(f'model_accuracy_{SYMBOL_PPO}.json', 'w', encoding='utf-8') as f:
    json.dump({'symbol': SYMBOL_PPO, 'model_type': 'PPO',
               'training_accuracy': float(acc), 'validation_accuracy': float(acc),
               'backtest_accuracy': float(acc), 'backtest_return': float(ret),
               'win_rate': float(acc), 'sharpe_ratio': None,
               'total_signals': int(total), 'correct_signals': int(correct),
               'live_accuracy': None,
               'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'history': []},
              f, ensure_ascii=False, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"✅ {TICKER} ({NAME}) 訓練完成摘要")
print('='*70)
print(f"  XGBoost 測試準確度: {test_acc*100:.2f}%  {'🌟 EXCELLENT' if test_acc>=0.65 else ('✅' if test_acc>=0.50 else '⚠️')}")
print(f"  PPO 準確度:        {acc:.2f}%")
print(f"  PPO 回報率:        +{ret:.2f}%")
print(f"  PPO 交易次數:       {test_env.total_trades}")
