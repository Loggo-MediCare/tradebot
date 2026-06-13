"""


8927.TWO (北基) XGBoost 交易信號生成器


"""


import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import sys


import io


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO


import pandas as pd


import yfinance as yf
from datetime import datetime


from model_accuracy_tracker import get_model_accuracy_display
from tw_news_tracker import print_tavily_news_tw


TICKER = '8927.TWO'


MODEL_FILE = 'ppo_8927_two_improved'


def add_technical_indicators(df):


    """添加技術指標"""


    df['sma_10'] = df['close'].rolling(10).mean()


    df['sma_30'] = df['close'].rolling(30).mean()


    df['sma_50'] = df['close'].rolling(50).mean()


    df['sma_200'] = df['close'].rolling(200).mean()


    df['ema_12'] = df['close'].ewm(span=12).mean()


    df['ema_26'] = df['close'].ewm(span=26).mean()


    delta = df['close'].diff()


    gain = delta.where(delta > 0, 0).rolling(14).mean()


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


    return df.bfill().ffill()



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
        obs = np.array([
            float(self.shares_held), float(self.balance), float(row['close']),
            float(row.get('sma_10', 0)), float(row.get('sma_30', 0)), float(row.get('sma_50', 0)),
            float(row.get('rsi', 50)), float(row.get('macd', 0)), float(row.get('macd_signal', 0)),
            float(row.get('bb_upper', 0)), float(row.get('bb_lower', 0)), float(row.get('volume', 0)),
            float(self.total_profit), float(stock_ratio), float(cash_ratio)
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
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_observation(), float(reward), done, False, {}


def get_trading_signal():


    """生成今日交易信號"""


    # 壓縮標題區塊


    accuracy_display = get_model_accuracy_display(TICKER)


    print(f"🤖 {TICKER} (北基) | 準確度: {accuracy_display} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")


    print("=" * 80)


    # 1. 加載 PPO 模型


    try:


        model = PPO.load(MODEL_FILE)


        print(f"✅ 模型加載成功: {MODEL_FILE}")


    except Exception as e:


        print(f"❌ 模型加載失敗: {e}")


        return None


    # 2. 下載最新數據


    print(f"\n📊 下載 {TICKER} 最新數據...")


    try:


        df = yf.download(TICKER, period='1y', progress=False)


        if df.empty:


            print("❌ 無法獲取數據")


            return None


        if isinstance(df.columns, pd.MultiIndex):


            df.columns = df.columns.droplevel(1)


        df = df.rename(columns={


            'Close': 'close', 'Volume': 'volume',


            'Open': 'open', 'High': 'high', 'Low': 'low'


        }).reset_index()


        print(f"✅ 成功下載 {len(df)} 天數據")


    except Exception as e:


        print(f"❌ 數據下載失敗: {e}")


        return None


    # 3. 添加技術指標


    df = add_technical_indicators(df)


    # 4. 獲取最新數據


    latest = df.iloc[-1]


    current_price = float(latest['close'])


    feature_columns = [


        'rsi', 'macd', 'macd_signal', 'macd_hist',


        'bb_position', 'K', 'D', 'obv', 'obv_ma20',


        'sma_10', 'sma_30', 'sma_50', 'sma_200',


        'volatility', 'atr',


        'price_change_5d', 'price_change_10d', 'price_change_20d',


        'ma50_slope'


    ]


    # PPO 環境與預測
    env = ImprovedTradingEnv(df)
    env.current_step = len(df) - 1
    obs = env._get_observation()

    print("\n🧠 PPO AI 模型分析中...")
    action, _ = model.predict(obs, deterministic=True)
    action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
    print(f"   PPO Action Value: {action_value:+.4f}")

    rsi = float(latest.get('rsi', 50)); macd = float(latest.get('macd', 0)); ms = float(latest.get('macd_signal', 0))
    s10 = float(latest.get('sma_10', 0)); s30 = float(latest.get('sma_30', 0))

    print("\n" + "=" * 80 + "\n📊 技術指標\n" + "=" * 80)
    print(f"當前價格: NT${current_price:.2f}")
    print(f"RSI: {rsi:.2f}  " + ("[超買]" if rsi > 70 else "[超賣]" if rsi < 30 else "[中性]"))
    print(f"MACD: {macd:.4f}  " + ("[金叉]" if macd > ms else "[死叉]"))
    print(f"均線: SMA10={s10:.2f}, SMA30={s30:.2f}  " + ("[多頭]" if s10 > s30 else "[空頭]"))

    try:
        patterns = analyze_candlestick_patterns(df, days=5)
        print(format_pattern_output(patterns))
        print("\n型態評分調整: " + f"{get_pattern_score_adjustment(patterns):+.1f}" + " 分")
    except Exception:
        pass

    print("\n" + "=" * 80 + "\n🎯 交易信號\n" + "=" * 80)
    print(f"模型輸出動作值: {action_value:+.4f}")
    if action_value > 0.1:
        print("🟢 買入信號 (BUY)")
        print(f"   PPO 動作強度: {action_value:+.4f}")
        print(f"   建議買入價格: NT${current_price*0.995:.2f} - NT${current_price*1.005:.2f}")
    elif action_value < -0.1:
        print("🔴 不建議買入 (SELL/WAIT)")
        print(f"   PPO 動作強度: {action_value:+.4f}")
    else:
        print("🟡 持有 (HOLD)")
        print(f"   PPO 動作強度: {action_value:+.4f}")

    print("=" * 80)
    return {'ticker': TICKER, 'price': current_price, 'action_value': action_value, 'rsi': rsi, 'macd': macd}


if __name__ == "__main__":


    get_trading_signal()



# ── Tavily 即時新聞 ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n' + '=' * 80)
    print('🌐 8927 北基 即時新聞  (Tavily REST API)')
    print('=' * 80)
    print_tavily_news_tw('8927', '北基', max_results=5)