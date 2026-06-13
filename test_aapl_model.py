"""
测试 AAPL 真实股票模型
=====================
加载并测试基于真实 AAPL 数据训练的模型
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')

# 配置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 交易环境 (必须与训练时相同)
class SimpleTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(SimpleTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
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
            float(self.shares_held),
            float(self.balance),
            float(row['close']),
            float(row.get('sma_10', 0)),
            float(row.get('sma_30', 0)),
            float(row.get('rsi', 50)),
            float(row.get('macd', 0)),
            float(row.get('volume', 0)),
            float(self.total_profit),
            float(self.current_step) / float(len(self.df))
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
        obs = self._get_observation()
        return obs, reward, done, False, {}

def add_technical_indicators(df):
    """添加技术指标"""
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2

    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

print("=" * 70)
print("📈 测试 AAPL 股票交易模型")
print("=" * 70)

# 1. 加载模型
model_path = "C:/Users/Silvi/ppo_aapl_model.zip"
print(f"\n加载模型: {model_path}")

try:
    model = PPO.load(model_path)
    print("✅ 模型加载成功!")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit(1)

# 2. 下载测试数据 (最新的 AAPL 数据)
print("\n下载最新 AAPL 测试数据...")
try:
    import yfinance as yf

    # 下载 2024 年的数据 (模型没见过的新数据)
    test_df = yf.download('AAPL', start='2024-01-01', end='2024-12-18', progress=False)

    test_df = test_df.rename(columns={
        'Close': 'close', 'Volume': 'volume',
        'Open': 'open', 'High': 'high', 'Low': 'low'
    })
    test_df = test_df.reset_index()

    print(f"✅ 下载了 {len(test_df)} 天的数据")
    print(f"   日期范围: {test_df['Date'].iloc[0]} 至 {test_df['Date'].iloc[-1]}")

except Exception as e:
    print(f"❌ 下载失败: {e}")
    exit(1)

# 3. 添加技术指标
test_df = add_technical_indicators(test_df)

# 4. 运行测试
print("\n" + "=" * 70)
print("开始测试...")
print("=" * 70)

env = SimpleTradingEnv(test_df)
obs, _ = env.reset()
done = False

actions_history = []
prices_history = []
total_assets_history = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    actions_history.append(action)
    prices_history.append(env.df.iloc[env.current_step - 1]['close'])
    total_asset = env.balance + env.shares_held * env.df.iloc[env.current_step - 1]['close']
    total_assets_history.append(total_asset)

# 5. 计算结果
final_profit = float(env.total_profit)
final_return = (final_profit / env.initial_balance) * 100
buy_hold_return = ((float(prices_history[-1]) - float(prices_history[0])) / float(prices_history[0])) * 100

print("\n" + "=" * 70)
print("📊 测试结果")
print("=" * 70)
print(f"初始资金:        ${env.initial_balance:,.2f}")
print(f"最终资产:        ${env.initial_balance + final_profit:,.2f}")
print(f"总收益:          ${final_profit:,.2f}")
print(f"AI 策略收益率:   {final_return:.2f}%")
print(f"买入持有收益率:  {buy_hold_return:.2f}%")
print(f"超额收益:        {final_return - buy_hold_return:.2f}%")

actions_count = {
    0: actions_history.count(0),
    1: actions_history.count(1),
    2: actions_history.count(2)
}
total = len(actions_history)
print(f"\n交易决策分布:")
print(f"  卖出: {actions_count[0]} 次 ({actions_count[0]/total*100:.1f}%)")
print(f"  持有: {actions_count[1]} 次 ({actions_count[1]/total*100:.1f}%)")
print(f"  买入: {actions_count[2]} 次 ({actions_count[2]/total*100:.1f}%)")

# 6. 可视化
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 价格和买卖点
ax1 = axes[0]
ax1.plot(prices_history, label='AAPL 股价', color='black', linewidth=2)

buy_points = [i for i, a in enumerate(actions_history) if a == 2]
sell_points = [i for i, a in enumerate(actions_history) if a == 0]

if buy_points:
    ax1.scatter(buy_points, [prices_history[i] for i in buy_points],
               color='green', marker='^', s=100, label='买入', zorder=5)
if sell_points:
    ax1.scatter(sell_points, [prices_history[i] for i in sell_points],
               color='red', marker='v', s=100, label='卖出', zorder=5)

ax1.set_title('AAPL 价格走势和 AI 交易决策 (2024年)', fontsize=16, fontweight='bold')
ax1.set_ylabel('股价 ($)', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# 总资产变化
ax2 = axes[1]
ax2.plot(total_assets_history, label='AI 策略总资产', color='blue', linewidth=2)
ax2.axhline(y=10000, color='gray', linestyle='--', label='初始资金', linewidth=1.5)

# 买入持有策略
buy_hold_assets = [10000 * (1 + (p - prices_history[0]) / prices_history[0]) for p in prices_history]
ax2.plot(buy_hold_assets, label='买入持有策略', color='orange', linewidth=2, alpha=0.7)

ax2.set_title('投资组合价值对比', fontsize=16, fontweight='bold')
ax2.set_xlabel('交易日', fontsize=12)
ax2.set_ylabel('总资产 ($)', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('aapl_ai_trading_results.png', dpi=150, bbox_inches='tight')
print(f"\n📊 图表已保存: aapl_ai_trading_results.png")
plt.show()

print("\n✅ 测试完成!")
