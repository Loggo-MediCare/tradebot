"""
使用真实股票数据训练 PPO 模型
=====================================
支持: AAPL, TSLA, MSFT, GOOGL, 等任意股票
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings
warnings.filterwarnings('ignore')

# 配置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 下载真实股票数据
# ==========================================
def download_stock_data(ticker='AAPL', start_date='2020-01-01', end_date='2024-01-01'):
    """
    下载真实股票数据

    参数:
        ticker: 股票代码 (AAPL, TSLA, MSFT, GOOGL, etc.)
        start_date: 开始日期
        end_date: 结束日期
    """
    print("=" * 60)
    print(f"下载 {ticker} 股票数据...")
    print("=" * 60)

    try:
        import yfinance as yf

        # 下载数据
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if df.empty:
            raise ValueError(f"无法下载 {ticker} 的数据")

        # 重命名列 (yfinance 使用大写)
        df = df.rename(columns={
            'Close': 'close',
            'Volume': 'volume',
            'Open': 'open',
            'High': 'high',
            'Low': 'low'
        })

        # 重置索引
        df = df.reset_index()
        df = df[['Date', 'close', 'volume', 'open', 'high', 'low']]
        df.columns = ['date', 'close', 'volume', 'open', 'high', 'low']

        print(f"✅ 成功下载 {len(df)} 天的数据")
        print(f"   日期范围: {df['date'].iloc[0]} 至 {df['date'].iloc[-1]}")
        print(f"   价格范围: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

        return df

    except ImportError:
        print("❌ 未安装 yfinance 库")
        print("   请运行: pip install yfinance")
        return None
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None

# ==========================================
# 2. 添加技术指标
# ==========================================
def add_technical_indicators(df):
    """添加技术指标"""
    print("\n添加技术指标...")

    # SMA - 简单移动平均
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    # EMA - 指数移动平均
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    # RSI - 相对强弱指标
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

    # 成交量指标
    df['volume_sma'] = df['volume'].rolling(20).mean()

    # 填充 NaN
    df = df.fillna(method='bfill').fillna(method='ffill')

    print(f"✅ 添加了 {len(['sma_10', 'sma_30', 'rsi', 'macd'])} 个技术指标")

    return df

# ==========================================
# 3. 交易环境 (与之前相同)
# ==========================================
class SimpleTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(SimpleTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(10,), dtype=np.float32
        )
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
        obs = np.array([
            self.shares_held, self.balance, row['close'],
            row.get('sma_10', 0), row.get('sma_30', 0),
            row.get('rsi', 50), row.get('macd', 0),
            row.get('volume', 0), self.total_profit,
            self.current_step / len(self.df)
        ], dtype=np.float32)
        return obs

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']

        if action == 0 and self.shares_held > 0:
            self.balance += self.shares_held * current_price
            self.shares_held = 0
        elif action == 2:
            shares_can_buy = self.balance // current_price
            if shares_can_buy > 0:
                self.shares_held += shares_can_buy
                self.balance -= shares_can_buy * current_price

        total_asset = self.balance + self.shares_held * current_price
        self.total_profit = total_asset - self.initial_balance
        reward = self.total_profit / self.initial_balance

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = False
        obs = self._get_observation()

        return obs, reward, done, truncated, {}

# ==========================================
# 4. 训练模型
# ==========================================
def train_model(df, ticker, total_timesteps=50000):
    print("\n" + "=" * 60)
    print(f"开始训练 {ticker} 交易模型")
    print("=" * 60)

    env = DummyVecEnv([lambda: SimpleTradingEnv(df)])

    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )

    print(f"\n训练参数:")
    print(f"  总步数: {total_timesteps}")
    print(f"  数据点: {len(df)}")
    print(f"  学习率: 0.0003")

    print("\n开始训练...")
    model.learn(total_timesteps=total_timesteps)

    model_path = f"ppo_{ticker.lower()}_model"
    model.save(model_path)
    print(f"\n✅ 模型已保存: {model_path}.zip")

    return model

# ==========================================
# 5. 测试模型
# ==========================================
def test_model(model, df, ticker):
    print("\n" + "=" * 60)
    print(f"测试 {ticker} 模型")
    print("=" * 60)

    env = SimpleTradingEnv(df)
    obs, _ = env.reset()
    done = False

    actions_history = []
    prices_history = []
    dates_history = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        actions_history.append(action)
        prices_history.append(env.df.iloc[env.current_step - 1]['close'])
        dates_history.append(env.df.iloc[env.current_step - 1]['date'])

    final_profit = env.total_profit
    final_return = (final_profit / env.initial_balance) * 100

    print(f"\n测试结果:")
    print(f"  初始资金: ${env.initial_balance:,.2f}")
    print(f"  最终资产: ${env.initial_balance + final_profit:,.2f}")
    print(f"  总收益:   ${final_profit:,.2f}")
    print(f"  收益率:   {final_return:.2f}%")

    # 计算基准收益 (买入持有策略)
    buy_hold_return = ((prices_history[-1] - prices_history[0]) / prices_history[0]) * 100
    print(f"\n对比:")
    print(f"  AI策略收益率:    {final_return:.2f}%")
    print(f"  买入持有收益率:  {buy_hold_return:.2f}%")
    print(f"  超额收益:        {final_return - buy_hold_return:.2f}%")

    return {
        'profit': final_profit,
        'return': final_return,
        'actions': actions_history,
        'prices': prices_history,
        'dates': dates_history
    }

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 配置
    TICKER = 'AAPL'  # 🔧 修改这里选择股票: AAPL, TSLA, MSFT, GOOGL, etc.
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'
    TRAIN_TEST_SPLIT = 0.8  # 80% 训练, 20% 测试

    print("🚀 真实股票交易 AI 训练系统")
    print("=" * 60)
    print(f"目标股票: {TICKER}")
    print(f"日期范围: {START_DATE} - {END_DATE}")
    print("=" * 60)

    # 1. 下载数据
    df = download_stock_data(TICKER, START_DATE, END_DATE)

    if df is None:
        print("\n❌ 无法继续,请先安装 yfinance:")
        print("   pip install yfinance")
        exit(1)

    # 2. 添加技术指标
    df = add_technical_indicators(df)

    # 3. 分割训练/测试数据
    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"\n数据分割:")
    print(f"  训练集: {len(train_df)} 天 ({train_df['date'].iloc[0]} - {train_df['date'].iloc[-1]})")
    print(f"  测试集: {len(test_df)} 天 ({test_df['date'].iloc[0]} - {test_df['date'].iloc[-1]})")

    # 4. 训练模型
    model = train_model(train_df, TICKER, total_timesteps=30000)

    # 5. 测试模型
    results = test_model(model, test_df, TICKER)

    print("\n✅ 完成!")
    print(f"模型文件: ppo_{TICKER.lower()}_model.zip")
