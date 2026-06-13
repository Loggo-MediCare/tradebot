"""
批量趨勢跟蹤模型訓練
====================
使用不依賴 RSI 的趨勢跟蹤策略批量訓練股票

核心改進：
1. 不使用 RSI 數值作為賣出依據
2. 上升趨勢中強制持有
3. 獎勵函數鼓勵順勢操作
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ==========================================
# 趨勢跟蹤交易環境 (不用 RSI)
# ==========================================
class TrendFollowingEnv(gym.Env):
    """趨勢跟蹤環境 - 不依賴 RSI"""

    def __init__(self, df, initial_balance=10000):
        super(TrendFollowingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0
        self.total_trades = 0
        self.consecutive_hold = 0
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        current_price = float(row['close'])
        total_value = self.balance + self.shares_held * current_price
        stock_ratio = (self.shares_held * current_price) / total_value if total_value > 0 else 0

        ma10 = float(row.get('sma_10', current_price))
        ma30 = float(row.get('sma_30', current_price))
        ma50 = float(row.get('sma_50', current_price))

        price_vs_ma10 = (current_price - ma10) / ma10 * 100 if ma10 > 0 else 0
        price_vs_ma30 = (current_price - ma30) / ma30 * 100 if ma30 > 0 else 0
        price_vs_ma50 = (current_price - ma50) / ma50 * 100 if ma50 > 0 else 0
        ma_trend = 1.0 if ma10 > ma30 > ma50 else (-1.0 if ma10 < ma30 < ma50 else 0.0)

        macd = float(row.get('macd', 0))
        macd_signal = float(row.get('macd_signal', 0))
        macd_hist = float(row.get('macd_hist', 0))
        macd_trend = 1.0 if macd > macd_signal else -1.0

        volume_ratio = float(row.get('volume_ratio', 1.0))
        price_change = float(row.get('price_change_1d', 0))
        volume_price_sync = 1.0 if (price_change > 0 and volume_ratio > 1.2) else (
            -1.0 if (price_change < 0 and volume_ratio > 1.2) else 0.0
        )

        atr_pct = float(row.get('atr_pct', 2.0))
        bb_position = float(row.get('bb_position', 50))
        ma50_slope = float(row.get('ma50_slope', 0))

        obs = np.array([
            float(self.shares_held) / 1000,
            float(self.balance) / self.initial_balance,
            float(stock_ratio),
            float(current_price) / 1000,
            float(price_vs_ma10),
            float(price_vs_ma30),
            float(price_vs_ma50),
            float(ma_trend),
            float(ma50_slope),
            float(row.get('price_momentum', 0)),
            float(macd) * 10,
            float(macd_signal) * 10,
            float(macd_hist) * 10,
            float(macd_trend),
            float(volume_ratio),
            float(volume_price_sync),
            float(atr_pct),
            float(bb_position) / 100,
        ], dtype=np.float32)

        return obs

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = float(action[0])
        action = np.clip(action, -1.0, 1.0)

        row = self.df.iloc[self.current_step]
        current_price = float(row['close'])

        ma10 = float(row.get('sma_10', current_price))
        ma30 = float(row.get('sma_30', current_price))
        ma50 = float(row.get('sma_50', current_price))
        macd = float(row.get('macd', 0))
        macd_signal = float(row.get('macd_signal', 0))

        is_uptrend = (ma10 > ma30) and (macd > macd_signal) and (current_price > ma50)
        is_downtrend = (ma10 < ma30) and (macd < macd_signal) and (current_price < ma50)

        # 趨勢跟蹤交易邏輯
        if action < -0.1:
            if is_uptrend:
                action = 0  # 上升趨勢不賣
                self.consecutive_hold += 1
            else:
                sell_ratio = abs(action)
                shares_to_sell = int(self.shares_held * sell_ratio)
                if shares_to_sell > 0:
                    self.balance += shares_to_sell * current_price
                    self.shares_held -= shares_to_sell
                    self.total_trades += 1
                    self.consecutive_hold = 0
        elif action > 0.1:
            if is_downtrend:
                action = action * 0.3
            buy_ratio = abs(action)
            max_can_buy = int(self.balance // current_price)
            shares_to_buy = int(max_can_buy * buy_ratio)
            if shares_to_buy > 0:
                self.balance -= shares_to_buy * current_price
                self.shares_held += shares_to_buy
                self.total_trades += 1
                self.consecutive_hold = 0
        else:
            self.consecutive_hold += 1

        new_total_value = self.balance + self.shares_held * current_price
        self.total_profit = new_total_value - self.initial_balance

        # 趨勢獎勵
        profit_reward = self.total_profit / self.initial_balance
        trend_reward = 0.0
        if is_uptrend and self.shares_held > 0:
            trend_reward = 0.02
        elif is_downtrend and self.shares_held == 0:
            trend_reward = 0.01
        elif is_uptrend and self.shares_held == 0:
            trend_reward = -0.02
        elif is_downtrend and self.shares_held > 0:
            trend_reward = -0.01

        hold_bonus = min(self.consecutive_hold * 0.001, 0.01)
        reward = profit_reward + trend_reward + hold_bonus

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_observation(), float(reward), done, False, {}


# ==========================================
# 數據處理
# ==========================================
def download_data(ticker, start_date, end_date):
    try:
        import yfinance as yf
        yf.set_tz_cache_location(r"C:\Users\Silvi\Projects\trading-bot\TMP")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            df = yf.download(ticker, period="max", progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.rename(columns={'Close': 'close', 'Volume': 'volume', 'Open': 'open', 'High': 'high', 'Low': 'low'})
        return df.reset_index()
    except:
        return None


def add_indicators(df):
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = ((df['close'] - df['bb_lower']) / bb_range * 100).fillna(50)

    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    df['price_change_1d'] = df['close'].pct_change(1) * 100
    df['price_momentum'] = df['close'].pct_change(5) * 100
    df['ma50_slope'] = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100

    # 未來漲跌方向 (用於特徵重要性分析)
    df['future_direction'] = (df['close'].shift(-5) > df['close']).astype(int)

    return df.fillna(method='bfill').fillna(method='ffill')


def analyze_trend_features(df, ticker):
    """分析趨勢指標重要性 (不含 RSI)"""
    # 趨勢跟蹤特徵 (不含 RSI)
    features = [
        'macd', 'macd_signal', 'macd_hist',
        'bb_position', 'volume_ratio',
        'price_momentum', 'ma50_slope', 'atr_pct',
        'price_change_1d'
    ]

    ml_data = df.dropna(subset=features + ['future_direction'])
    if len(ml_data) < 100:
        return None

    X = ml_data[features]
    y = ml_data['future_direction']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    importances = rf_model.feature_importances_
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 保存 JSON
    json_data = {
        'ticker': ticker,
        'model_type': 'trend_following_v2',
        'analysis_date': datetime.now().strftime('%Y-%m-%d'),
        'model_accuracy': float(accuracy),
        'note': 'RSI removed from features - trend following model',
        'feature_importance': {
            features[i]: float(importances[i])
            for i in range(len(features))
        }
    }

    json_filename = f'{ticker.replace(".", "_")}_trend_v2_feature_importance.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    return accuracy, json_filename


def train_model(df, ticker, timesteps=100000):
    env = DummyVecEnv([lambda: TrendFollowingEnv(df)])
    model = PPO('MlpPolicy', env, verbose=0, learning_rate=0.0003, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01)
    model.learn(total_timesteps=timesteps)
    model_path = f"ppo_{ticker.lower().replace('.', '_')}_trend_v2"
    model.save(model_path)
    return model_path


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("批量趨勢跟蹤模型訓練 (不依賴 RSI)")
    print("=" * 70)

    # 要訓練的股票列表
    STOCKS = [
        # 台股
        {'ticker': '2317.TW', 'name': '鴻海'},
        {'ticker': '2454.TW', 'name': '聯發科'},
        {'ticker': '2308.TW', 'name': '台達電'},
        {'ticker': '3008.TW', 'name': '大立光'},
        {'ticker': '2881.TW', 'name': '富邦金'},
        {'ticker': '1513.TW', 'name': '中興電'},
        {'ticker': '1519.TW', 'name': '華城'},
        {'ticker': '8996.TW', 'name': '高力'},
        # 美股
        {'ticker': 'NVDA', 'name': 'NVIDIA'},
        {'ticker': 'TSLA', 'name': 'Tesla'},
        {'ticker': 'AMD', 'name': 'AMD'},
        {'ticker': 'AAPL', 'name': 'Apple'},
    ]

    START_DATE = '2015-01-01'
    END_DATE = '2026-12-31'
    TIMESTEPS = 100000

    success = []
    failed = []

    for stock in STOCKS:
        ticker = stock['ticker']
        name = stock['name']

        print(f"\n{'='*50}")
        print(f"訓練: {ticker} ({name})")
        print(f"{'='*50}")

        # 下載數據
        df = download_data(ticker, START_DATE, END_DATE)
        if df is None or len(df) < 200:
            print(f"  [跳過] 數據不足")
            failed.append(f"{ticker} ({name})")
            continue

        print(f"  數據: {len(df)} 天")

        # 添加指標
        df = add_indicators(df)

        # 訓練
        split_idx = int(len(df) * 0.8)
        train_df = df[:split_idx].copy()

        print(f"  訓練集: {len(train_df)} 天")
        print(f"  訓練中... ({TIMESTEPS:,} 步)")

        try:
            # 訓練模型
            model_path = train_model(train_df, ticker, TIMESTEPS)
            print(f"  [模型] {model_path}.zip")

            # 特徵重要性分析
            result = analyze_trend_features(df, ticker)
            if result:
                accuracy, json_file = result
                print(f"  [分析] {json_file} (準確率: {accuracy:.1%})")

            success.append(f"{ticker} ({name})")
        except Exception as e:
            print(f"  [失敗] {e}")
            failed.append(f"{ticker} ({name})")

    # 總結
    print("\n" + "=" * 70)
    print("訓練完成!")
    print("=" * 70)
    print(f"\n成功: {len(success)} 支")
    for s in success:
        print(f"  ✅ {s}")

    if failed:
        print(f"\n失敗: {len(failed)} 支")
        for f in failed:
            print(f"  ❌ {f}")

    print("\n核心特點:")
    print("  1. 不使用 RSI 作為賣出依據")
    print("  2. 上升趨勢中強制持有")
    print("  3. 順勢獎勵函數")
