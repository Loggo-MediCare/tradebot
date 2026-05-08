import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 改進的交易環境（關鍵升級）
# ==========================================
class AdvancedTradingEnv(gym.Env):
    """
    升級版交易環境
    - 30天時間窗口觀察
    - 連續動作空間
    - 真實交易成本
    - 風險調整獎勵
    """
    metadata = {"render_modes": []}

    def __init__(self, df, initial_balance=100000, window_size=30):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.current_step = 0

        # 動作空間：-1.0（全賣）~ +1.0（全買）
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 觀察空間：30天 × 特徵數（扁平化）
        num_features = len(self._get_features(df.iloc[0]))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size * num_features,), dtype=np.float32
        )

        self.reset()

    def _get_features(self, row):
        """定義單日特徵（可擴充）"""
        return np.array([
            row['close'],
            row.get('sma_10', 0), row.get('sma_30', 0), row.get('sma_50', 0),
            row.get('rsi', 50), row.get('macd', 0), row.get('macd_signal', 0),
            row.get('bb_upper', 0), row.get('bb_lower', 0),
            row.get('K', 50), row.get('D', 50),  # KD
            row.get('OBV', 0), row.get('ATR', 0),
            row.get('volatility', 0),
            row.get('price_change_5d', 0), row.get('price_change_20d', 0),
            row.get('volume', 0)
        ], dtype=np.float32)

    def _get_observation(self):
        start = max(self.current_step - self.window_size + 1, 0)
        window = self.df.iloc[start:self.current_step + 1]
        # 填充不足部分
        if len(window) < self.window_size:
            pad = np.repeat([window.iloc[0].values], self.window_size - len(window), axis=0)
            window = pd.DataFrame(np.vstack([pad, window.values]))
        features = np.hstack([self._get_features(row) for _, row in window.iterrows()])
        return features.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size - 1  # 確保有足夠歷史
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = 0
        self.returns_history = []
        return self._get_observation(), {}

    def step(self, action):
        action = float(np.clip(action, -1.0, 1.0))
        current_price = float(self.df.iloc[self.current_step]['close'])

        # 交易成本
        commission_rate = 0.001425  # 手續費
        tax_rate = 0.003           # 賣出證交稅

        if action > 0.1:  # 買入
            buy_ratio = action
            max_buy = self.balance / current_price * (1 - commission_rate)
            shares_to_buy = int(max_buy * buy_ratio)
            cost = shares_to_buy * current_price * (1 + commission_rate)
            if cost <= self.balance:
                self.balance -= cost
                self.shares_held += shares_to_buy
                self.trades += 1

        elif action < -0.1:  # 賣出
            sell_ratio = abs(action)
            shares_to_sell = int(self.shares_held * sell_ratio)
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price * (1 - commission_rate - tax_rate)
                self.balance += revenue
                self.shares_held -= shares_to_sell
                self.trades += 1

        # 更新淨值
        prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # 計算日收益
        step_return = (self.net_worth - prev_net_worth) / prev_net_worth if prev_net_worth > 0 else 0
        self.returns_history.append(step_return)

        # 風險調整獎勵（類 Sortino）
        profit_reward = step_return
        downside_std = np.std([r for r in self.returns_history[-60:] if r < 0]) if len(self.returns_history) >= 10 else 0
        risk_penalty = -2.0 * downside_std if downside_std > 0 else 0
        trade_bonus = 0.02 if abs(action) > 0.1 else 0  # 適度鼓勵交易

        reward = profit_reward + risk_penalty + trade_bonus

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        return self._get_observation(), float(reward), done, False, {}

# ==========================================
# 2. 數據準備（保留你原有的指標）
# ==========================================
def add_technical_indicators(df):
    # （你的原函數，略微補強）
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    df['bb_middle'] = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * std
    df['bb_lower'] = df['bb_middle'] - 2 * std
    
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['K'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
    df['D'] = df['K'].rolling(3).mean()
    
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    df['price_change_5d'] = df['close'].pct_change(5)
    df['price_change_20d'] = df['close'].pct_change(20)
    
    df = df.fillna(method='bfill').fillna(0)
    return df

# ==========================================
# 3. 訓練模型（關鍵升級）
# ==========================================
def train_advanced_model(ticker='2331.TW', total_timesteps=500000):
    print(f"\n🚀 開始訓練 {ticker} 進階交易 AI...")
    
    # 下載數據
    df = yf.download(ticker, start='2015-01-01', end='2025-12-31', progress=False)
    if df.empty:
        raise ValueError("無法下載數據")
    df = df.rename(columns=lambda x: x.lower())
    df = add_technical_indicators(df)
    
    # 環境 + 正規化
    env = DummyVecEnv([lambda: AdvancedTradingEnv(df)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    # LSTM Policy
    policy_kwargs = dict(
        features_extractor_class=BaseFeaturesExtractor,
        net_arch=[256, 256],
        activation_fn=nn.Tanh,
    )
    
    model = PPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard/"
    )
    
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"PPO_{ticker}_LSTM")
    
    model_path = f"ppo_{ticker.replace('.', '_')}_advanced"
    model.save(model_path)
    env.save(f"{model_path}_vecnormalize.pkl")
    
    print(f"✅ 模型訓練完成並已保存：{model_path}.zip")
    return model

# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    TICKER = '2331.TW'  # 或 '2330.TW'
    train_advanced_model(TICKER, total_timesteps=500000)