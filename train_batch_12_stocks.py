"""
批量训练 12 只台股 PPO 模型
=====================================
股票列表: 2327, 1303, 8046, 6239, 1301, 6488, 1802, 2360, 2313, 8996, 3443, 6446
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
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==========================================
# 要训练的股票列表
# ==========================================
STOCKS_TO_TRAIN = [
    '3374.TW',
    '5347.TW',
    '8210.TW',
    '8080.TW',
    '2344.TW',
    '2360.TW',
    '2330.TW',
    '3260.TW',
    '8299.TW',
    '6446.TW',  # 藥華藥 (retry)
]

# ==========================================
# 交易环境
# ==========================================
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
        self.last_action = 0
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        current_price = float(row['close'])
        total_value = self.balance + self.shares_held * current_price
        stock_ratio = (self.shares_held * current_price) / total_value if total_value > 0 else 0
        cash_ratio = self.balance / total_value if total_value > 0 else 1
        obs = np.array([
            float(self.shares_held),
            float(self.balance),
            float(row['close']),
            float(row.get('sma_10', 0)),
            float(row.get('sma_30', 0)),
            float(row.get('sma_50', 0)),
            float(row.get('rsi', 50)),
            float(row.get('macd', 0)),
            float(row.get('macd_signal', 0)),
            float(row.get('bb_upper', 0)),
            float(row.get('bb_lower', 0)),
            float(row.get('volume', 0)),
            float(self.total_profit),
            float(stock_ratio),
            float(cash_ratio),
        ], dtype=np.float32)
        return obs

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = float(action[0])
        else:
            action = float(action)
        action = np.clip(action, -1.0, 1.0)
        current_price = float(self.df.iloc[self.current_step]['close'])
        old_total_value = self.balance + self.shares_held * current_price

        if action < -0.1:
            sell_ratio = abs(action)
            shares_to_sell = int(self.shares_held * sell_ratio)
            if shares_to_sell > 0:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                self.total_trades += 1
        elif action > 0.1:
            buy_ratio = action
            max_can_buy = int(self.balance // current_price)
            shares_to_buy = int(max_can_buy * buy_ratio)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.balance -= cost
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
        obs = self._get_observation()
        return obs, float(reward), done, False, {}

# ==========================================
# 数据处理函数
# ==========================================
def download_and_prepare_data(ticker, start_date='2015-01-01', end_date='2025-01-01'):
    print(f"\n下载 {ticker} 数据...")
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty:
            raise ValueError(f"无法下载 {ticker}")
        df = df.rename(columns={
            'Close': 'close', 'Volume': 'volume',
            'Open': 'open', 'High': 'high', 'Low': 'low'
        })
        df = df.reset_index()
        print(f"  ✅ {len(df)} 天数据")
        return df
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        return None

def add_technical_indicators(df):
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = ((df['close'] - df['bb_lower']) / bb_range * 100).fillna(50)
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['K'] = ((df['close'] - low_14) / (high_14 - low_14) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(20).mean()
    df['MA_20'] = df['close'].rolling(20).mean()
    df['MA_50'] = df['close'].rolling(50).mean()
    df['MA_200'] = df['close'].rolling(200).mean()
    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['price_change_5d'] = df['close'].pct_change(5) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100
    df['MA50_slope'] = df['MA_50'].diff(5) / df['MA_50'].shift(5) * 100
    df['future_direction'] = (df['close'].shift(-5) > df['close']).astype(int)
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

def analyze_feature_importance(df, ticker):
    features = [
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_position', 'K', 'D', 'OBV', 'OBV_MA',
        'MA_20', 'MA_50', 'MA_200',
        'volatility', 'ATR', 'price_change_5d', 'price_change_20d', 'MA50_slope'
    ]
    ml_data = df.dropna(subset=features + ['future_direction'])
    if len(ml_data) == 0:
        return None
    X = ml_data[features]
    y = ml_data['future_direction']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  特徵分析準確率: {accuracy:.2%}")

    # 保存 JSON
    import json
    from datetime import datetime
    json_data = {
        'ticker': ticker,
        'analysis_date': datetime.now().strftime('%Y-%m-%d'),
        'model_accuracy': float(accuracy),
        'feature_importance': {
            row['Feature']: float(row['Importance'])
            for _, row in feature_importance_df.iterrows()
        }
    }
    json_filename = f'{ticker.replace(".", "_")}_feature_importance.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    return feature_importance_df

def train_model(df, ticker, total_timesteps=100000):
    env = DummyVecEnv([lambda: ImprovedTradingEnv(df)])
    model = PPO(
        'MlpPolicy', env, verbose=0,
        learning_rate=0.0003, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=0.99, ent_coef=0.01,
    )
    model.learn(total_timesteps=total_timesteps)
    model_path = f"ppo_{ticker.lower().replace('.', '_')}_improved"
    model.save(model_path)
    print(f"  ✅ 模型已保存: {model_path}.zip")
    return model

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 批量训练 12 只台股 PPO 模型")
    print("=" * 70)

    START_DATE = '2015-01-01'
    END_DATE = '2025-07-31'
    TOTAL_TIMESTEPS = 100000

    success_count = 0
    failed_stocks = []

    for i, ticker in enumerate(STOCKS_TO_TRAIN, 1):
        print(f"\n[{i}/{len(STOCKS_TO_TRAIN)}] 训练 {ticker}")
        print("-" * 50)

        # 1. 下载数据
        df = download_and_prepare_data(ticker, START_DATE, END_DATE)
        if df is None:
            failed_stocks.append(ticker)
            continue

        # 2. 添加技术指标
        df = add_technical_indicators(df)

        # 3. 特征分析
        analyze_feature_importance(df, ticker)

        # 4. 分割数据
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()

        # 5. 训练模型
        print(f"  训练中 (100,000 步)...")
        train_model(train_df, ticker, total_timesteps=TOTAL_TIMESTEPS)
        success_count += 1

    print("\n" + "=" * 70)
    print("📊 训练完成统计")
    print("=" * 70)
    print(f"成功: {success_count}/{len(STOCKS_TO_TRAIN)}")
    if failed_stocks:
        print(f"失败: {', '.join(failed_stocks)}")
    print("=" * 70)
