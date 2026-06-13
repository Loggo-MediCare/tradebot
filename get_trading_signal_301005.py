"""


301005.SZ AI 交易信号生成器


使用训练好的 PPO 模型生成今日交易策略


"""


import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


os.environ['MPLBACKEND'] = 'Agg'  # Fix Tcl/Tk error on Windows


import sys, io


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


import numpy as np


import pandas as pd


import gymnasium as gym


from gymnasium import spaces


from stable_baselines3 import PPO


from datetime import datetime


import warnings
from tw_news_tracker import print_tavily_news_tw


warnings.filterwarnings('ignore')


TICKER = '301005.SZ'


MODEL_PATH = r"C:\Users\Silvi\Projects\trading-bot\ppo_301005_sz_improved"


class TradingEnv(gym.Env):


    def __init__(self, df):


        super().__init__()


        self.df = df.reset_index(drop=True)


        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)


        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)


        self.reset()


    def reset(self, seed=None, options=None):


        super().reset(seed=seed)


        self.step_idx = 0


        self.balance = 10000


        self.shares = 0


        self.profit = 0


        return self._obs(), {}


    def _obs(self):


        r = self.df.iloc[self.step_idx]


        p = float(r['close'])


        t = self.balance + self.shares * p


        return np.array([self.shares, self.balance, p,


            float(r.get('sma_10',0)), float(r.get('sma_30',0)), float(r.get('sma_50',0)),


            float(r.get('rsi',50)), float(r.get('macd',0)), float(r.get('macd_signal',0)),


            float(r.get('bb_upper',0)), float(r.get('bb_lower',0)), float(r.get('volume',0)),


            self.profit, (self.shares*p)/t if t>0 else 0, self.balance/t if t>0 else 1], dtype=np.float32)


    def step(self, action):


        a = np.clip(float(action[0]) if isinstance(action, np.ndarray) else float(action), -1, 1)


        p = float(self.df.iloc[self.step_idx]['close'])


        if a < -0.1 and self.shares > 0:


            s = int(self.shares * abs(a))


            self.balance += s * p


            self.shares -= s


        elif a > 0.1 and self.balance > p:


            b = int((self.balance // p) * a)


            self.balance -= b * p


            self.shares += b


        self.profit = self.balance + self.shares * p - 10000


        self.step_idx += 1


        return self._obs(), self.profit/10000, self.step_idx >= len(self.df)-1, False, {}


def add_indicators(df):


    df['sma_10'] = df['close'].rolling(10).mean()


    df['sma_30'] = df['close'].rolling(30).mean()


    df['sma_50'] = df['close'].rolling(50).mean()


    df['ema_12'] = df['close'].ewm(span=12).mean()


    df['ema_26'] = df['close'].ewm(span=26).mean()


    d = df['close'].diff()


    df['rsi'] = 100 - (100/(1+(d.where(d>0,0)).rolling(14).mean()/((-d.where(d<0,0)).rolling(14).mean()+1e-10)))


    df['macd'] = df['ema_12'] - df['ema_26']


    df['macd_signal'] = df['macd'].ewm(span=9).mean()


    m = df['close'].rolling(20).mean()


    s = df['close'].rolling(20).std()


    df['bb_upper'] = m + s*2


    df['bb_lower'] = m - s*2


    return df.bfill().ffill()


def get_trading_signal():


    print("=" * 60)


    print(f"🤖 {TICKER} AI 交易信号生成器")


    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


    print("=" * 60)


    # 加载模型


    try:


        model = PPO.load(MODEL_PATH)


        print("✅ 模型加载成功")


    except Exception as e:


        print(f"❌ 模型加载失败: {e}")


        return None


    # 下载数据


    try:


        import yfinance as yf


        df = yf.download(TICKER, period='90d', progress=False, auto_adjust=True)


        if isinstance(df.columns, pd.MultiIndex):


            df.columns = df.columns.droplevel(1)


        if df.empty:


            print("❌ 无法获取数据")


            return None


        df = df.rename(columns={'Close':'close','Volume':'volume','Open':'open','High':'high','Low':'low'})


        df = df.reset_index()


        print(f"✅ 下载 {len(df)} 天数据")


    except Exception as e:


        print(f"❌ 数据下载失败: {e}")


        return None


    df = add_indicators(df)


    latest = df.iloc[-1]


    price = float(latest['close'])


    rsi = float(latest['rsi'])


    macd = float(latest['macd'])


    macd_sig = float(latest['macd_signal'])


    sma_10 = float(latest['sma_10'])


    sma_30 = float(latest['sma_30'])


    bb_upper = float(latest['bb_upper'])


    bb_lower = float(latest['bb_lower'])


    vol = float(latest['volume'])


    avg_vol = float(df['volume'].tail(20).mean())


    vol_ratio = vol / avg_vol if avg_vol > 0 else 1


    print(f"\n📊 技术指标")


    print(f"价格: ${price:.2f}")


    print(f"RSI: {rsi:.1f} {'[超买]' if rsi>70 else '[超卖]' if rsi<30 else ''}")


    print(f"MACD: {macd:.4f} / {macd_sig:.4f} {'[金叉]' if macd>macd_sig else '[死叉]'}")


    print(f"SMA: {sma_10:.2f} / {sma_30:.2f} {'[多头]' if sma_10>sma_30 else '[空头]'}")


    print(f"量比: {vol_ratio:.2f}x")


    # 预测


    env = TradingEnv(df)


    env.step_idx = len(df) - 1


    obs = env._obs()


    action, _ = model.predict(obs, deterministic=True)


    action_val = float(action[0]) if isinstance(action, np.ndarray) else float(action)


    print(f"\n🎯 AI 信号")


    print(f"动作值: {action_val:+.4f}")


    if action_val > 0.1:


        signal = "买入 (BUY)"


        emoji = "🟢"


    elif action_val < -0.1:


        signal = "卖出 (SELL)"


        emoji = "🔴"


    else:


        signal = "持有 (HOLD)"


        emoji = "🟡"


    print(f"{emoji} 信号: {signal}")


    print(f"强度: {abs(action_val):.2f}")


    print("\n" + "=" * 60)


    print("⚠️ 风险提示: 本信号仅供参考,不构成投资建议")


    print("=" * 60)


if __name__ == "__main__":


    result = get_trading_signal()


    if result:


        print(f"\n✅ {result['symbol']} @ ${result['price']:.2f} -> {result['signal']}")



# ── Tavily 即時新聞 ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n' + '=' * 80)
    print('🌐 301005 301005 即時新聞  (Tavily REST API)')
    print('=' * 80)
    print_tavily_news_tw('301005', '301005', max_results=5)