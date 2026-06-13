"""
批量训练第6批股票 PPO 模型
=====================================
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

STOCKS_TO_TRAIN = [
    '7717.TW',   # 利多
    '6510.TW',   # 精測
    '4768.TW',   # 晶呈科技
    '2344.TW',   # 華邦電
    '2317.TW',   # 鴻海
    '2382.TW',   # 廣達
    '2408.TW',   # 南亞科
    '2383.TW',   # 台光電
    '2308.TW',   # 台達電
    '2505.TW',   # 國泰建設
    '3036.TW',   # 文曄
    '3057.TW',   # 喬鼎
    '3017.TW',   # 奇鋐
    '3609.TW',   # 緯創資通
    '3231.TW',   # 緯創
    '3661.TW',   # 世芯-KY
    '8131.TW',   # 福懋科
    '8110.TW',   # 華東
    '7769.TW',   # 聚和
]

class ImprovedTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(ImprovedTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0
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
            float(row.get('sma_10', 0)), float(row.get('sma_30', 0)), float(row.get('sma_50', 0)),
            float(row.get('rsi', 50)), float(row.get('macd', 0)), float(row.get('macd_signal', 0)),
            float(row.get('bb_upper', 0)), float(row.get('bb_lower', 0)), float(row.get('volume', 0)),
            float(self.total_profit), float(stock_ratio), float(cash_ratio),
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
        reward = self.total_profit / self.initial_balance
        reward += 0.01 if abs(action) > 0.1 else 0.0
        reward -= 0.005 if self.balance > old_total_value * 0.9 else 0.0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_observation(), float(reward), done, False, {}

def download_data(ticker):
    print(f"\n下载 {ticker} ...")
    try:
        import yfinance as yf
        df = yf.download(ticker, start='2015-01-01', end='2025-07-31', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty:
            raise ValueError(f"无法下载 {ticker}")
        df = df.rename(columns={'Close': 'close', 'Volume': 'volume', 'Open': 'open', 'High': 'high', 'Low': 'low'})
        df = df.reset_index()
        print(f"  ✅ {len(df)} 天")
        return df
    except Exception as e:
        print(f"  ❌ {e}")
        return None

def add_indicators(df):
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['future_direction'] = (df['close'].shift(-5) > df['close']).astype(int)
    return df.fillna(method='bfill').fillna(method='ffill')

def analyze_features(df, ticker):
    features = ['rsi', 'macd', 'macd_signal', 'sma_10', 'sma_30', 'sma_50']
    ml_data = df.dropna(subset=features + ['future_direction'])
    if len(ml_data) == 0:
        return
    X = ml_data[features]
    y = ml_data['future_direction']
    X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2, shuffle=False)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"  準確率: {acc:.1%}")
    import json
    from datetime import datetime
    with open(f'{ticker.replace(".", "_")}_feature_importance.json', 'w') as f:
        json.dump({'ticker': ticker, 'date': datetime.now().strftime('%Y-%m-%d'), 'accuracy': float(acc)}, f)

def train_model(df, ticker):
    env = DummyVecEnv([lambda: ImprovedTradingEnv(df)])
    model = PPO('MlpPolicy', env, verbose=0, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01)
    model.learn(total_timesteps=100000)
    model_path = f"ppo_{ticker.lower().replace('.', '_')}_improved"
    model.save(model_path)
    print(f"  ✅ {model_path}.zip")

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 批量训练第6批 (19 stocks)")
    print("=" * 50)

    success, failed = 0, []
    for i, ticker in enumerate(STOCKS_TO_TRAIN, 1):
        print(f"\n[{i}/{len(STOCKS_TO_TRAIN)}] {ticker}")
        df = download_data(ticker)
        if df is None:
            failed.append(ticker)
            continue
        df = add_indicators(df)
        analyze_features(df, ticker)
        train_df = df.iloc[:int(len(df) * 0.8)].copy()
        train_model(train_df, ticker)
        success += 1

    print(f"\n{'=' * 50}")
    print(f"✅ {success}/{len(STOCKS_TO_TRAIN)}")
    if failed:
        print(f"❌ {', '.join(failed)}")
