"""
批量训练台股 AI 交易模型
========================
自动训练以下股票:
- 4938 和碩
- 6443 元晶
- 6209 今國光
- 2449 京元電子
- 5498 凱崴
- 3017 奇鋐
- 6805 富世達
- 8210 勤誠
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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# 股票列表
STOCKS = [
    {'code': 'INTC', 'name': 'Intel', 'file': 'intc'},
    {'code': 'MSFT', 'name': 'Microsoft', 'file': 'msft'},
    {'code': 'AMZN', 'name': 'Amazon', 'file': 'amzn'},
    {'code': 'META', 'name': 'Meta', 'file': 'meta'},
    {'code': 'NFLX', 'name': 'Netflix', 'file': 'nflx'},
    {'code': 'QCOM', 'name': 'Qualcomm', 'file': 'qcom'},
    {'code': 'ASML', 'name': 'ASML', 'file': 'asml'},
    {'code': 'ARM', 'name': 'ARM Holdings', 'file': 'arm'},
    {'code': 'SMCI', 'name': 'Super Micro', 'file': 'smci'},
    {'code': 'SNPS', 'name': 'Synopsys', 'file': 'snps'},
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
        reward = self.total_profit / self.initial_balance

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self._get_observation()

        return obs, float(reward), done, False, {}

# ==========================================
# 技术指标
# ==========================================
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
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

    df = df.bfill().ffill()
    return df

# ==========================================
# 训练单个股票
# ==========================================
def train_stock(stock_code, stock_name, file_code):
    print("\n" + "=" * 80)
    print(f"开始训练: {stock_code} ({stock_name})")
    print("=" * 80)

    # 1. 下载数据
    print(f"\n📊 下载 {stock_name} 数据...")
    try:
        df = yf.download(stock_code, start='2015-01-01', end='2024-12-31', progress=False)

        if df.empty:
            print(f"❌ {stock_name} 数据下载失败，跳过")
            return False

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.rename(columns={
            'Close': 'close', 'Volume': 'volume',
            'Open': 'open', 'High': 'high', 'Low': 'low'
        })
        df = df.reset_index()

        print(f"✅ 成功下载 {len(df)} 天数据 ({df.iloc[0]['Date'].date()} 到 {df.iloc[-1]['Date'].date()})")

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

    # 2. 添加技术指标
    print(f"\n🔧 计算技术指标...")
    df = add_technical_indicators(df)
    print(f"✅ 技术指标计算完成")

    # 3. 创建环境
    print(f"\n🏗️  创建训练环境...")
    env = DummyVecEnv([lambda: ImprovedTradingEnv(df)])
    print(f"✅ 环境创建完成")

    # 4. 训练模型
    print(f"\n🤖 开始训练 PPO 模型...")
    print(f"   训练步数: 100,000")

    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0
    )

    try:
        model.learn(total_timesteps=100000)
        print(f"✅ 模型训练完成!")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return False

    # 5. 保存模型
    model_path = f"ppo_{file_code}_improved.zip"
    model.save(model_path)
    print(f"\n💾 模型已保存: {model_path}")

    # 6. 测试模型
    print(f"\n📈 测试模型表现...")
    obs = env.reset()
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward[0]

    env_instance = env.envs[0]
    final_value = env_instance.balance + env_instance.shares_held * df.iloc[-1]['close']
    profit = final_value - env_instance.initial_balance
    profit_pct = (profit / env_instance.initial_balance) * 100

    print(f"✅ 测试完成!")
    print(f"   初始资金: NT${env_instance.initial_balance:,.0f}")
    print(f"   最终价值: NT${final_value:,.0f}")
    print(f"   总收益: NT${profit:,.0f} ({profit_pct:+.2f}%)")
    print(f"   交易次数: {env_instance.total_trades}")

    return True

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 批量训练台股 AI 交易模型")
    print("=" * 80)
    print(f"总共需要训练 {len(STOCKS)} 只股票")

    success_count = 0
    failed_stocks = []

    for i, stock in enumerate(STOCKS, 1):
        print(f"\n进度: [{i}/{len(STOCKS)}]")

        success = train_stock(stock['code'], stock['name'], stock['file'])

        if success:
            success_count += 1
        else:
            failed_stocks.append(f"{stock['code']} ({stock['name']})")

    # 最终总结
    print("\n" + "=" * 80)
    print("🎉 批量训练完成!")
    print("=" * 80)
    print(f"✅ 成功训练: {success_count}/{len(STOCKS)}")
    print(f"❌ 失败数量: {len(failed_stocks)}")

    if failed_stocks:
        print(f"\n失败的股票:")
        for stock in failed_stocks:
            print(f"   • {stock}")

    print("\n所有训练完成! 🎊")
