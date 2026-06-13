"""


6488.TWO (環球晶) PPO 交易信號生成器 — PPO 69.19% beats XGBoost 63.93%


"""


import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import sys, io, warnings


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


warnings.filterwarnings('ignore')


import numpy as np


import pandas as pd


import yfinance as yf


import gymnasium as gym


from gymnasium import spaces


from stable_baselines3 import PPO


from datetime import datetime


from model_accuracy_tracker import get_model_accuracy_display
from tw_news_tracker import print_tavily_news_tw


TICKER     = '6488.TWO'


MODEL_FILE = 'ppo_6488_two_improved'


NAME       = '環球晶'


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


        return self._get_observation(), {}


    def _get_observation(self):


        row = self.df.iloc[self.current_step]


        price = float(row['close'])


        total_value = self.balance + self.shares_held * price


        return np.array([


            float(self.shares_held), float(self.balance), price,


            float(row.get('sma_10', 0)), float(row.get('sma_30', 0)),


            float(row.get('sma_50', 0)), float(row.get('rsi', 50)),


            float(row.get('macd', 0)), float(row.get('macd_signal', 0)),


            float(row.get('bb_upper', 0)), float(row.get('bb_lower', 0)),


            float(row.get('volume', 0)), float(self.total_profit),


            (self.shares_held * price) / total_value if total_value > 0 else 0,


            self.balance / total_value if total_value > 0 else 1,


        ], dtype=np.float32)


    def step(self, action):


        action = float(action[0]) if isinstance(action, np.ndarray) else float(action)


        action = np.clip(action, -1.0, 1.0)


        price = float(self.df.iloc[self.current_step]['close'])


        if action < -0.1:


            shares = int(self.shares_held * abs(action))


            if shares > 0:


                self.balance += shares * price; self.shares_held -= shares


        elif action > 0.1:


            shares = int((self.balance // price) * action)


            if shares > 0:


                self.balance -= shares * price; self.shares_held += shares


        self.total_profit = (self.balance + self.shares_held * price) - self.initial_balance


        self.current_step += 1


        done = self.current_step >= len(self.df) - 1


        return self._get_observation(), self.total_profit / self.initial_balance, done, False, {}


def add_technical_indicators(df):


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


    df['bb_middle'] = df['close'].rolling(20).mean()


    df['bb_std'] = df['close'].rolling(20).std()


    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']


    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']


    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100).fillna(50)


    lo14 = df['low'].rolling(14).min()


    hi14 = df['high'].rolling(14).max()


    df['K'] = ((df['close'] - lo14) / (hi14 - lo14) * 100).fillna(50)


    df['D'] = df['K'].rolling(3).mean()


    return df.bfill().ffill()


def get_trading_signal():


    accuracy_display = get_model_accuracy_display(TICKER)


    print(f"🤖 {TICKER} ({NAME}) | 準確度: {accuracy_display} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")


    print("=" * 80)


    try:


        model = PPO.load(MODEL_FILE)


        print(f"✅ PPO 模型加載成功: {MODEL_FILE}.zip")


    except Exception as e:


        print(f"❌ 模型加載失敗: {e}"); return None


    print(f"\n📊 下載 {TICKER} 最新數據...")


    try:


        df = yf.download(TICKER, period='300d', progress=False, auto_adjust=True)


        if df.empty:


            print("❌ 無法獲取數據"); return None


        if isinstance(df.columns, pd.MultiIndex):


            df.columns = df.columns.droplevel(1)


        df = df.rename(columns={'Close':'close','Volume':'volume','Open':'open','High':'high','Low':'low'}).reset_index()


        print(f"✅ 成功下載 {len(df)} 天數據")


    except Exception as e:


        print(f"❌ 數據下載失敗: {e}"); return None


    df = add_technical_indicators(df)


    latest = df.iloc[-1]


    prev_close   = float(df['close'].iloc[-2])


    current_price = float(latest['close'])


    price_change_pct = (current_price - prev_close) / prev_close * 100


    env = TradingEnv(df)


    env.current_step = len(df) - 1


    obs = env._get_observation()


    print("\n🧠 AI 模型分析中...")


    action, _ = model.predict(obs, deterministic=True)


    action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)


    rsi  = float(latest['rsi']); macd = float(latest['macd']); ms = float(latest['macd_signal'])


    s10  = float(latest['sma_10']); s30 = float(latest['sma_30'])


    bb_upper = float(latest['bb_upper']); bb_lower = float(latest['bb_lower'])


    avg_vol = float(df['volume'].tail(20).mean())


    vol_ratio = float(latest['volume']) / avg_vol if avg_vol > 0 else 1.0


    candle_dir = 'up' if current_price > prev_close else 'down' if current_price < prev_close else 'flat'


    print("\n" + "=" * 80)


    print("📊 技術指標")


    print("=" * 80)


    print(f"當前價格:        NT${current_price:.2f}")


    print(f"今日漲跌幅:      {price_change_pct:+.2f}%")


    print(f"RSI: {rsi:.2f}  {'[超買]' if rsi > 70 else '[超賣]' if rsi < 30 else '[中性]'}")


    print(f"MACD: {macd:.4f}  {'[金叉]' if macd > ms else '[死叉]'}")


    print(f"均線: SMA10={s10:.2f}, SMA30={s30:.2f}  {'[多頭]' if s10 > s30 else '[空頭]'}")


    print(f"布林帶位置:      {float(latest['bb_position']):.1f}%")


    print(f"KD: K={float(latest['K']):.1f}, D={float(latest['D']):.1f}")


    print(f"量比:            {vol_ratio:.2f}x  {'[放量]' if vol_ratio > 1.5 else '[縮量]' if vol_ratio < 0.7 else '[正常]'}")


    print(f"量價方向:        {'價漲量增' if candle_dir == 'up' and vol_ratio >= 1.2 else '價跌量增' if candle_dir == 'down' and vol_ratio >= 1.2 else '中性'}")


    try:


        patterns = analyze_candlestick_patterns(df, days=5)


        print(format_pattern_output(patterns))


        print(f"\n型態評分調整: {get_pattern_score_adjustment(patterns):+.1f} 分")


    except Exception:


        pass


    print("\n" + "=" * 80)


    print("🎯 交易信號 (PPO)")


    print("=" * 80)


    print(f"模型動作值: {action_value:+.4f}")


    if action_value > 0.1:


        print("🟢 買入信號 (BUY)")


        print(f"   AI 強度: {action_value:.2f} / 1.00")


        print(f"   建議買入價格: NT${current_price*0.995:.2f} - NT${current_price*1.000:.2f}")


        print(f"   止損參考: NT${current_price*0.95:.2f} (-5%)")


    elif action_value < -0.1:


        print("🔴 賣出/觀望 (SELL/WAIT)")


        print(f"   AI 強度: {abs(action_value):.2f} / 1.00")


    else:


        print("🟡 持有 (HOLD)")


        print(f"   關注支撐位: NT${bb_lower:.2f}")


        print(f"   關注壓力位: NT${bb_upper:.2f}")


    print("=" * 80)


    return {'ticker': TICKER, 'price': current_price,


            'signal': 'BUY' if action_value > 0.1 else 'SELL' if action_value < -0.1 else 'HOLD',


            'action_value': action_value, 'rsi': rsi, 'macd': macd}


if __name__ == "__main__":


    get_trading_signal()



# ── Tavily 即時新聞 ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n' + '=' * 80)
    print('🌐 6488 環球晶 即時新聞  (Tavily REST API)')
    print('=' * 80)
    print_tavily_news_tw('6488', '環球晶', max_results=5)