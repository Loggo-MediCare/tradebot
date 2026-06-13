"""
Simplified FinRL Example (No TensorTrade Required)
=========================================
Using Stable Baselines3 + Custom FinRL-style Environment
Avoiding complex dependency installation issues
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



# ==========================================
# 1. 创建交易环境 (基于 Gymnasium)
# ==========================================
class SimpleTradingEnv(gym.Env):
    """简化的交易环境 - 兼容 Stable Baselines3"""

    def __init__(self, df, initial_balance=10000):
        super(SimpleTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0

        # 动作空间: 0=卖出, 1=持有, 2=买入
        self.action_space = spaces.Discrete(3)

        # 观察空间: [持有股票数量, 账户余额, 当前价格, ...技术指标]
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
        """获取当前状态"""
        row = self.df.iloc[self.current_step]

        obs = np.array([
            self.shares_held,           # 持有股票数量
            self.balance,               # 账户余额
            row['close'],               # 当前价格
            row.get('sma_10', 0),       # 10日均线
            row.get('sma_30', 0),       # 30日均线
            row.get('rsi', 50),         # RSI指标
            row.get('macd', 0),         # MACD
            row.get('volume', 0),       # 成交量
            self.total_profit,          # 总收益
            self.current_step / len(self.df)  # 进度
        ], dtype=np.float32)

        return obs

    def step(self, action):
        """执行动作"""
        current_price = self.df.iloc[self.current_step]['close']

        # 执行交易
        if action == 0:  # 卖出
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0

        elif action == 2:  # 买入
            shares_can_buy = self.balance // current_price
            if shares_can_buy > 0:
                self.shares_held += shares_can_buy
                self.balance -= shares_can_buy * current_price

        # 计算当前总资产
        total_asset = self.balance + self.shares_held * current_price
        self.total_profit = total_asset - self.initial_balance

        # 奖励 = 收益率
        reward = self.total_profit / self.initial_balance

        # 移动到下一步
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = False

        if done:
            obs = self._get_observation()
        else:
            obs = self._get_observation()

        return obs, reward, done, truncated, {}

    def render(self):
        profit = self.total_profit
        print(f"Step: {self.current_step}, Profit: ${profit:.2f}")

# ==========================================
# 2. 生成模拟数据 (或加载真实数据)
# ==========================================
def generate_sample_data(n_days=365):
    """生成模拟股价数据"""
    np.random.seed(42)

    # 生成价格
    price = 100
    prices = []
    for _ in range(n_days):
        change = np.random.randn() * 2
        price = max(price + change, 10)
        prices.append(price)

    df = pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_days)
    })

    # 添加技术指标
    df['sma_10'] = df['close'].rolling(10).mean().fillna(df['close'])
    df['sma_30'] = df['close'].rolling(30).mean().fillna(df['close'])

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)

    # MACD
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd'] = df['macd'].fillna(0)

    return df

# ==========================================
# 3. 训练模型
# ==========================================
def train_model(df, total_timesteps=50000):
    """使用 PPO 算法训练交易智能体"""
    print("=" * 50)
    print("开始训练交易智能体")
    print("=" * 50)

    # 创建环境
    env = DummyVecEnv([lambda: SimpleTradingEnv(df)])

    # 创建 PPO 模型
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="./ppo_trading_tensorboard/"
    )

    # 训练
    print("\n开始训练...")
    model.learn(total_timesteps=total_timesteps)

    # 保存模型
    model.save("ppo_trading_model")
    print("\n✅ 模型已保存至: ppo_trading_model.zip")

    return model

# ==========================================
# 4. 测试模型
# ==========================================
def test_model(df, model_path="ppo_trading_model"):
    """测试训练好的模型"""
    print("\n" + "=" * 50)
    print("测试交易智能体")
    print("=" * 50)

    # 加载模型
    model = PPO.load(model_path)

    # 创建测试环境
    env = SimpleTradingEnv(df)

    # 运行测试
    obs, _ = env.reset()
    done = False
    total_reward = 0

    actions_history = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        actions_history.append(action)

    final_profit = env.total_profit
    final_return = (final_profit / env.initial_balance) * 100

    print(f"\n测试结果:")
    print(f"初始资金: ${env.initial_balance:.2f}")
    print(f"最终资产: ${env.initial_balance + final_profit:.2f}")
    print(f"总收益: ${final_profit:.2f}")
    print(f"收益率: {final_return:.2f}%")

    return final_profit, actions_history

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 生成数据
    print("生成模拟数据...")
    df = generate_sample_data(n_days=500)

    # 分割训练/测试数据
    train_df = df.iloc[:400]
    test_df = df.iloc[400:]

    # 2. 训练模型
    model = train_model(train_df, total_timesteps=20000)

    # 3. 测试模型
    profit, actions = test_model(test_df)

    # 4. 可视化
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_df['close'], label='训练数据价格')
    plt.title('训练数据')
    plt.xlabel('时间步')
    plt.ylabel('价格')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_df['close'].values, label='测试数据价格')
    plt.title('test data')
    plt.xlabel('timestep')
    plt.ylabel('price')
    plt.legend()

    plt.tight_layout()
    plt.savefig('trading_results.png')
    print("\n📊 图表已保存至: trading_results.png")
    plt.show()

    print("\n✅ 完成!")
