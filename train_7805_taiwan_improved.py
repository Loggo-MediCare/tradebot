"""
Taiwan Stock 7805 Trading AI Training
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ImprovedTradingEnv(gym.Env):
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
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        price = float(row['close'])
        total = self.balance + self.shares_held * price
        return np.array([float(self.shares_held), float(self.balance), price,
            float(row.get('sma_10', 0)), float(row.get('sma_30', 0)), float(row.get('sma_50', 0)),
            float(row.get('rsi', 50)), float(row.get('macd', 0)), float(row.get('macd_signal', 0)),
            float(row.get('bb_upper', 0)), float(row.get('bb_lower', 0)), float(row.get('volume', 0)),
            float(self.total_profit), (self.shares_held * price) / total if total > 0 else 0,
            self.balance / total if total > 0 else 1], dtype=np.float32)

    def step(self, action):
        action = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        action = np.clip(action, -1.0, 1.0)
        price = float(self.df.iloc[self.current_step]['close'])
        if action < -0.1 and self.shares_held > 0:
            sell = int(self.shares_held * abs(action))
            if sell > 0: self.balance += sell * price; self.shares_held -= sell
        elif action > 0.1:
            buy = int((self.balance // price) * action)
            if buy > 0: self.balance -= buy * price; self.shares_held += buy
        self.total_profit = (self.balance + self.shares_held * price) - self.initial_balance
        reward = self.total_profit / self.initial_balance + (0.01 if abs(action) > 0.1 else 0)
        self.current_step += 1
        return self._get_observation(), float(reward), self.current_step >= len(self.df) - 1, False, {}

def download_data(ticker):
    import yfinance as yf
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start='2015-01-01', end='2025-07-31', progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    df = df.rename(columns={'Close': 'close', 'Volume': 'volume', 'Open': 'open', 'High': 'high', 'Low': 'low'}).reset_index()
    print(f"Downloaded {len(df)} days")
    return df

def add_indicators(df):
    df['sma_10'], df['sma_30'], df['sma_50'] = df['close'].rolling(10).mean(), df['close'].rolling(30).mean(), df['close'].rolling(50).mean()
    df['ema_12'], df['ema_26'] = df['close'].ewm(span=12).mean(), df['close'].ewm(span=26).mean()
    delta = df['close'].diff()
    df['rsi'] = 100 - (100 / (1 + (delta.where(delta > 0, 0)).rolling(14).mean() / ((-delta.where(delta < 0, 0)).rolling(14).mean() + 1e-10)))
    df['macd'], df['macd_signal'] = df['ema_12'] - df['ema_26'], (df['ema_12'] - df['ema_26']).ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['bb_middle'], df['bb_std'] = df['close'].rolling(20).mean(), df['close'].rolling(20).std()
    df['bb_upper'], df['bb_lower'] = df['bb_middle'] + 2*df['bb_std'], df['bb_middle'] - 2*df['bb_std']
    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100).fillna(50)
    low_14, high_14 = df['low'].rolling(14).min(), df['high'].rolling(14).max()
    df['K'] = ((df['close'] - low_14) / (high_14 - low_14) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(20).mean()
    df['MA_20'], df['MA_50'], df['MA_200'] = df['close'].rolling(20).mean(), df['close'].rolling(50).mean(), df['close'].rolling(200).mean()
    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    tr = pd.concat([df['high']-df['low'], np.abs(df['high']-df['close'].shift()), np.abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['price_change_5d'], df['price_change_20d'] = df['close'].pct_change(5)*100, df['close'].pct_change(20)*100
    df['MA50_slope'] = df['MA_50'].diff(5) / df['MA_50'].shift(5) * 100
    df['future_direction'] = (df['close'].shift(-5) > df['close']).astype(int)
    return df.fillna(method='bfill').fillna(method='ffill')

def analyze_features(df, ticker):
    print(f"\nFeature Importance Analysis")
    features = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position', 'K', 'D',
                'MA_20', 'volatility', 'ATR', 'price_change_5d']
    ml_data = df.dropna(subset=features + ['future_direction'])
    if len(ml_data) < 50:
        print(f"Insufficient data for feature analysis ({len(ml_data)} samples), skipping...")
        return
    X, y = ml_data[features], ml_data['future_direction']
    X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2, shuffle=False)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    imp_df = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)
    print(f"Accuracy: {acc:.4f}")
    print(imp_df.to_string(index=False))
    plt.figure(figsize=(10, 8))
    plt.barh(imp_df['Feature'], imp_df['Importance'], color='#3498DB')
    plt.xlabel("Importance"); plt.title(f"{ticker} Feature Importance")
    plt.gca().invert_yaxis(); plt.tight_layout()
    plt.savefig(f'{ticker}_feature_importance.png', dpi=300); plt.close()
    with open(f'{ticker}_feature_importance.json', 'w', encoding='utf-8') as f:
        json.dump({'ticker': ticker, 'date': datetime.now().strftime('%Y-%m-%d'), 'accuracy': float(acc),
                   'features': {r['Feature']: float(r['Importance']) for _, r in imp_df.iterrows()}}, f, indent=2)
    print(f"Saved: {ticker}_feature_importance.png/json")

if __name__ == "__main__":
    TICKER = '7805.TWO'
    print(f"Training {TICKER}")
    df = download_data(TICKER)
    if df is None or len(df) < 100:
        print("Insufficient data"); exit(1)
    df = add_indicators(df)
    analyze_features(df, TICKER.replace('.', '_'))
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    print(f"\nTrain: {len(train_df)} days, Test: {len(df)-split} days")
    env = DummyVecEnv([lambda: ImprovedTradingEnv(train_df)])
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01)
    model.learn(total_timesteps=100000)
    model.save(f"ppo_{TICKER.lower().replace('.', '_')}_improved")
    print(f"\nComplete! Model: ppo_{TICKER.lower().replace('.', '_')}_improved.zip")
