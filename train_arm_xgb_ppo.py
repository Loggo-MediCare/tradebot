"""
ARM (Arm Holdings) - Retrain XGBoost + PPO with fresh data through today.
Produces an accuracy comparison table between the two model types.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MPLBACKEND'] = 'Agg'
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

TICKER = 'ARM'
END_DATE = datetime.now().strftime('%Y-%m-%d')

print("=" * 80)
print(f"🚀 {TICKER} (Arm Holdings) - XGBoost + PPO 重新訓練")
print(f"   數據範圍: 2015-01-01 至 {END_DATE}")
print("=" * 80)

# ============================================================
# 1. 下載數據 + 技術指標
# ============================================================
df = yf.download(TICKER, start='2015-01-01', end=END_DATE, progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)
df = df.rename(columns={'Close': 'close', 'Volume': 'volume', 'Open': 'open',
                         'High': 'high', 'Low': 'low'}).reset_index()
print(f"✅ 下載 {len(df)} 天數據")
print(f"   價格範圍: ${float(df['close'].min()):.2f} - ${float(df['close'].max()):.2f}")

df['sma_10'] = df['close'].rolling(10).mean()
df['sma_30'] = df['close'].rolling(30).mean()
df['sma_50'] = df['close'].rolling(50).mean()
df['sma_200'] = df['close'].rolling(200).mean()
df['ema_12'] = df['close'].ewm(span=12).mean()
df['ema_26'] = df['close'].ewm(span=26).mean()
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
df['macd'] = df['ema_12'] - df['ema_26']
df['macd_signal'] = df['macd'].ewm(span=9).mean()
df['macd_hist'] = df['macd'] - df['macd_signal']
df['bb_middle'] = df['close'].rolling(20).mean()
df['bb_std'] = df['close'].rolling(20).std()
df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100).fillna(50)
low_14 = df['low'].rolling(14).min()
high_14 = df['high'].rolling(14).max()
df['K'] = ((df['close'] - low_14) / (high_14 - low_14) * 100).fillna(50)
df['D'] = df['K'].rolling(3).mean()
df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
df['obv_ma20'] = df['obv'].rolling(20).mean()
df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
high_low = df['high'] - df['low']
high_close = np.abs(df['high'] - df['close'].shift())
low_close = np.abs(df['low'] - df['close'].shift())
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['atr'] = true_range.rolling(14).mean()
df['price_change_5d'] = df['close'].pct_change(5) * 100
df['price_change_10d'] = df['close'].pct_change(10) * 100
df['price_change_20d'] = df['close'].pct_change(20) * 100
df['ma50_slope'] = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100
df['future_return'] = df['close'].shift(-5) / df['close'] - 1
df['target'] = (df['future_return'] > 0.02).astype(int)

xgb_features = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position', 'K', 'D',
                 'obv', 'obv_ma20', 'sma_10', 'sma_30', 'sma_50', 'sma_200',
                 'volatility', 'atr', 'price_change_5d', 'price_change_10d',
                 'price_change_20d', 'ma50_slope']

# ============================================================
# 2. XGBoost 訓練
# ============================================================
print("\n" + "=" * 80)
print("📊 XGBoost 模型訓練")
print("=" * 80)
df_clean = df.dropna(subset=xgb_features + ['target'])
X, y = df_clean[xgb_features], df_clean['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"訓練集: {len(X_train)}, 測試集: {len(X_test)}")

xgb_model = xgb.XGBClassifier(
    max_depth=5, learning_rate=0.05, n_estimators=200,
    min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
    objective='binary:logistic', random_state=42, eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

xgb_train_acc = accuracy_score(y_train, xgb_model.predict(X_train))
xgb_test_acc = accuracy_score(y_test, xgb_model.predict(X_test))
print(f"訓練準確度: {xgb_train_acc*100:.2f}%")
print(f"測試準確度: {xgb_test_acc*100:.2f}%")
print(classification_report(y_test, xgb_model.predict(X_test), target_names=['不買', '買入']))

joblib.dump(xgb_model, 'xgb_arm_model.pkl')
print("✅ XGBoost 模型已保存: xgb_arm_model.pkl")

with open('model_accuracy_ARM.json', 'w', encoding='utf-8') as f:
    json.dump({
        'symbol': 'ARM',
        'model_type': 'XGBoost',
        'training_accuracy': float(xgb_train_acc * 100),
        'validation_accuracy': float(xgb_test_acc * 100),
        'backtest_accuracy': float(xgb_test_acc * 100),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }, f, ensure_ascii=False, indent=2)
print("✅ 準確度已更新: model_accuracy_ARM.json")

# ============================================================
# 3. 特徵重要性 (供 dynamic_signal_weights 使用)
# ============================================================
print("\n" + "=" * 80)
print("📊 特徵重要性分析 (RandomForest)")
print("=" * 80)
fi_features = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position', 'K', 'D', 'obv', 'obv_ma20',
                'sma_10', 'sma_30', 'sma_50', 'sma_200', 'volatility', 'atr',
                'price_change_5d', 'price_change_10d', 'price_change_20d', 'ma50_slope']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[fi_features])
X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_tr, y_tr)
rf_acc = accuracy_score(y_te, rf.predict(X_te))
fi_df = pd.DataFrame({'Feature': fi_features, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)
print(f"RandomForest 測試準確度: {rf_acc*100:.2f}%")
print(fi_df.head(10).to_string(index=False))

with open('ARM_feature_importance.json', 'w', encoding='utf-8') as f:
    json.dump({
        'ticker': 'ARM',
        'analysis_date': datetime.now().strftime('%Y-%m-%d'),
        'model_accuracy': float(rf_acc),
        'feature_importance': {row['Feature']: float(row['Importance']) for _, row in fi_df.iterrows()}
    }, f, indent=2, ensure_ascii=False)
print("✅ 特徵重要性已保存: ARM_feature_importance.json")

# ============================================================
# 4. PPO 訓練 (15維觀測, 連續動作)
# ============================================================
class ImprovedTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(ImprovedTradingEnv, self).__init__()
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
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        current_price = float(row['close'])
        total_value = self.balance + self.shares_held * current_price
        stock_ratio = (self.shares_held * current_price) / total_value if total_value > 0 else 0
        cash_ratio = self.balance / total_value if total_value > 0 else 1
        return np.array([
            float(self.shares_held), float(self.balance), float(row['close']),
            float(row['sma_10']), float(row['sma_30']), float(row['sma_50']),
            float(row['rsi']), float(row['macd']), float(row['macd_signal']),
            float(row['bb_upper']), float(row['bb_lower']), float(row['volume']),
            float(self.total_profit), float(stock_ratio), float(cash_ratio)
        ], dtype=np.float32)

    def step(self, action):
        action = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        action = np.clip(action, -1.0, 1.0)
        current_price = float(self.df.iloc[self.current_step]['close'])
        old_total_value = self.balance + self.shares_held * current_price

        if action < -0.1:
            shares_to_sell = int(self.shares_held * abs(action))
            if shares_to_sell > 0:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                self.total_trades += 1
        elif action > 0.1:
            max_can_buy = int(self.balance // current_price)
            shares_to_buy = int(max_can_buy * action)
            if shares_to_buy > 0:
                self.balance -= shares_to_buy * current_price
                self.shares_held += shares_to_buy
                self.total_trades += 1

        new_total_value = self.balance + self.shares_held * current_price
        self.total_profit = new_total_value - self.initial_balance
        profit_reward = self.total_profit / self.initial_balance
        trade_incentive = 0.01 if abs(action) > 0.1 else 0.0
        cash_penalty = -0.005 if self.balance > old_total_value * 0.9 else 0.0
        reward = profit_reward + trade_incentive + cash_penalty

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_observation(), float(reward), done, False, {}


ppo_features = ['sma_10', 'sma_30', 'sma_50', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'close', 'volume']
df_ppo = df.bfill().ffill().dropna(subset=ppo_features)
split_idx = int(len(df_ppo) * 0.8)
train_df = df_ppo.iloc[:split_idx].reset_index(drop=True)
test_df = df_ppo.iloc[split_idx:].reset_index(drop=True)
print(f"\n" + "=" * 80)
print("🤖 PPO 模型訓練")
print("=" * 80)
print(f"訓練集: {len(train_df)} 天, 測試集: {len(test_df)} 天")

env = DummyVecEnv([lambda: ImprovedTradingEnv(train_df)])
model = PPO('MlpPolicy', env, learning_rate=0.0005, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, ent_coef=0.01, verbose=0)
print("訓練中... (200,000 步)")
model.learn(total_timesteps=200000)
model.save('ppo_arm_improved')
print("✅ PPO 模型已保存: ppo_arm_improved.zip")

# ============================================================
# 5. PPO 回測 (測試集)
# ============================================================
test_env = ImprovedTradingEnv(test_df)
obs, _ = test_env.reset()
correct = 0
total = 0
for _ in range(len(test_df) - 1):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = test_env.step(action)
    if abs(action[0]) > 0.15:
        total += 1
        correct += (1 if reward > 0 else 0)
    if done:
        break

ppo_acc = (correct / total * 100) if total > 0 else 0
final_value = test_env.balance + test_env.shares_held * test_df.iloc[-1]['close']
ppo_return = (final_value - 10000) / 10000 * 100
print(f"✅ PPO 回測準確度: {ppo_acc:.2f}% | 回報率: {ppo_return:.2f}% | 交易次數: {test_env.total_trades} | 樣本數: {total}")

with open('model_accuracy_ARM_ppo.json', 'w', encoding='utf-8') as f:
    json.dump({
        'symbol': 'ARM',
        'model_type': 'PPO',
        'training_accuracy': None,
        'validation_accuracy': None,
        'backtest_accuracy': float(ppo_acc),
        'backtest_return': float(ppo_return),
        'win_rate': float(ppo_acc),
        'sharpe_ratio': None,
        'total_signals': int(total),
        'correct_signals': int(correct),
        'live_accuracy': None,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'history': []
    }, f, ensure_ascii=False, indent=2)
print("✅ 準確度已更新: model_accuracy_ARM_ppo.json")

# ============================================================
# 6. 比較表
# ============================================================
print("\n" + "=" * 80)
print("📊 ARM 模型準確度比較")
print("=" * 80)
print(f"{'模型':<12} {'訓練準確度':>12} {'測試/回測準確度':>16} {'樣本數':>8}")
print(f"{'XGBoost':<12} {xgb_train_acc*100:>11.2f}% {xgb_test_acc*100:>15.2f}% {len(X_test):>8}")
print(f"{'PPO':<12} {'N/A':>12} {ppo_acc:>15.2f}% {total:>8}")
print("=" * 80)
print(f"PPO 回測期間回報率: {ppo_return:+.2f}% (測試集 {len(test_df)} 天)")
print("\n[OK] ARM 重新訓練完成!")
