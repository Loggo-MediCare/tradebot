"""
高级版 AAPL 股票交易 AI 训练
=====================================
进一步优化:
1. ✅ 添加卖出信号和止盈机制
2. ✅ 加入风险管理 (止损、波动率控制)
3. ✅ 使用 SAC 算法 (更适合连续动作)
4. ✅ 增强的奖励函数
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import SAC  # 🔥 使用 SAC 算法
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings
warnings.filterwarnings('ignore')

# 配置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==========================================
# 高级交易环境
# ==========================================
class AdvancedTradingEnv(gym.Env):
    """
    高级交易环境
    - 风险管理: 止损、止盈
    - 波动率惩罚
    - 鼓励适时卖出
    """

    def __init__(self, df, initial_balance=10000,
                 stop_loss_pct=0.1, take_profit_pct=0.2):
        super(AdvancedTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0

        # 🔥 风险管理参数
        self.stop_loss_pct = stop_loss_pct      # 止损: 10%
        self.take_profit_pct = take_profit_pct  # 止盈: 20%

        # 连续动作空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # 增强的观察空间 (18个特征)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(18,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0
        self.total_trades = 0
        self.buy_price = 0  # 记录买入价格
        self.max_portfolio_value = self.initial_balance  # 追踪最高资产
        self.consecutive_losses = 0  # 连续亏损次数
        return self._get_observation(), {}

    def _get_observation(self):
        """增强的观察空间"""
        row = self.df.iloc[self.current_step]
        current_price = float(row['close'])
        total_value = self.balance + self.shares_held * current_price

        # 持仓相关
        stock_ratio = (self.shares_held * current_price) / total_value if total_value > 0 else 0
        cash_ratio = self.balance / total_value if total_value > 0 else 1

        # 收益相关
        unrealized_pnl = (current_price - self.buy_price) / self.buy_price if self.buy_price > 0 else 0
        drawdown = (self.max_portfolio_value - total_value) / self.max_portfolio_value if self.max_portfolio_value > 0 else 0

        # 技术指标
        sma_10 = float(row.get('sma_10', current_price))
        sma_30 = float(row.get('sma_30', current_price))
        price_to_sma10 = (current_price - sma_10) / sma_10 if sma_10 > 0 else 0
        price_to_sma30 = (current_price - sma_30) / sma_30 if sma_30 > 0 else 0

        obs = np.array([
            float(self.shares_held),          # 0. 持股数量
            float(self.balance),              # 1. 现金余额
            current_price,                    # 2. 当前价格
            sma_10,                          # 3. SMA 10
            sma_30,                          # 4. SMA 30
            float(row.get('sma_50', current_price)),  # 5. SMA 50
            float(row.get('rsi', 50)),        # 6. RSI
            float(row.get('macd', 0)),        # 7. MACD
            float(row.get('macd_signal', 0)), # 8. MACD Signal
            float(row.get('bb_upper', current_price)),  # 9. Bollinger Upper
            float(row.get('bb_lower', current_price)),  # 10. Bollinger Lower
            float(row.get('volume', 0)),      # 11. 成交量
            float(self.total_profit),         # 12. 总收益
            float(stock_ratio),               # 13. 持股比例
            float(cash_ratio),                # 14. 现金比例
            float(unrealized_pnl),            # 15. 🔥 未实现盈亏
            float(drawdown),                  # 16. 🔥 回撤
            float(price_to_sma10),            # 17. 🔥 价格相对SMA10
        ], dtype=np.float32)

        return obs

    def step(self, action):
        """
        执行动作 (增强版)
        """
        if isinstance(action, np.ndarray):
            action = float(action[0])
        else:
            action = float(action)

        action = np.clip(action, -1.0, 1.0)
        current_price = float(self.df.iloc[self.current_step]['close'])
        old_total_value = self.balance + self.shares_held * current_price

        # 🔥 自动止损/止盈检查
        auto_sell = False
        if self.shares_held > 0 and self.buy_price > 0:
            current_return = (current_price - self.buy_price) / self.buy_price

            # 止损
            if current_return < -self.stop_loss_pct:
                action = -1.0  # 强制全部卖出
                auto_sell = True

            # 止盈
            elif current_return > self.take_profit_pct:
                action = -0.5  # 卖出一半锁定利润
                auto_sell = True

        # 执行交易
        if action < -0.1:  # 卖出
            sell_ratio = abs(action)
            shares_to_sell = int(self.shares_held * sell_ratio)

            if shares_to_sell > 0:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                self.total_trades += 1

                # 如果完全卖出,重置买入价格
                if self.shares_held == 0:
                    self.buy_price = 0

        elif action > 0.1:  # 买入
            buy_ratio = action
            max_can_buy = int(self.balance // current_price)
            shares_to_buy = int(max_can_buy * buy_ratio)

            if shares_to_buy > 0:
                cost = shares_to_buy * current_price

                # 记录买入价格(加权平均)
                if self.shares_held > 0 and self.buy_price > 0:
                    total_cost = self.shares_held * self.buy_price + cost
                    self.shares_held += shares_to_buy
                    self.buy_price = total_cost / self.shares_held
                else:
                    self.buy_price = current_price
                    self.shares_held = shares_to_buy

                self.balance -= cost
                self.total_trades += 1

        # 计算新的总价值和收益
        new_total_value = self.balance + self.shares_held * current_price
        self.total_profit = new_total_value - self.initial_balance

        # 更新最高资产值
        if new_total_value > self.max_portfolio_value:
            self.max_portfolio_value = new_total_value

        # 🔥 增强的奖励函数
        # 1. 基础收益奖励
        profit_reward = (new_total_value - old_total_value) / self.initial_balance

        # 2. 风险惩罚 - 惩罚过度集中
        risk_penalty = 0.0
        stock_ratio = (self.shares_held * current_price) / new_total_value if new_total_value > 0 else 0
        if stock_ratio > 0.9:  # 持股超过90%
            risk_penalty = -0.002
        elif stock_ratio < 0.1 and self.balance > new_total_value * 0.9:  # 现金超过90%
            risk_penalty = -0.001

        # 3. 卖出奖励 - 🔥 鼓励适时卖出
        sell_reward = 0.0
        if action < -0.1 and self.shares_held >= 0:  # 卖出动作
            sell_reward = 0.005  # 小额奖励

        # 4. 止损/止盈奖励
        auto_trade_reward = 0.0
        if auto_sell:
            auto_trade_reward = 0.01  # 奖励风险管理

        # 5. 回撤惩罚
        drawdown = (self.max_portfolio_value - new_total_value) / self.max_portfolio_value if self.max_portfolio_value > 0 else 0
        drawdown_penalty = -drawdown * 0.01

        # 综合奖励
        reward = (profit_reward +
                 risk_penalty +
                 sell_reward +
                 auto_trade_reward +
                 drawdown_penalty)

        # 移动到下一步
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self._get_observation()

        return obs, float(reward), done, False, {}

# ==========================================
# 数据处理函数
# ==========================================
def download_and_prepare_data(ticker='AAPL', start_date='2015-01-01', end_date='2024-01-01'):
    print("=" * 70)
    print(f"下载 {ticker} 股票数据...")
    print("=" * 70)

    try:
        import yfinance as yf
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if df.empty:
            raise ValueError(f"无法下载 {ticker} 的数据")

        df = df.rename(columns={
            'Close': 'close', 'Volume': 'volume',
            'Open': 'open', 'High': 'high', 'Low': 'low'
        })
        df = df.reset_index()

        print(f"✅ 成功下载 {len(df)} 天的数据")
        return df

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None

def add_technical_indicators(df):
    print("\n添加技术指标...")

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

    df = df.fillna(method='bfill').fillna(method='ffill')
    print(f"✅ 添加了多个技术指标")
    return df

# ==========================================
# 训练模型
# ==========================================
def train_advanced_model(df, ticker, total_timesteps=100000):
    """
    🔥 使用 SAC 算法训练
    """
    print("\n" + "=" * 70)
    print(f"开始训练高级版 {ticker} 交易模型")
    print("=" * 70)

    env = DummyVecEnv([lambda: AdvancedTradingEnv(df)])

    # 🔥 使用 SAC (Soft Actor-Critic)
    # SAC 更适合连续动作空间,探索能力更强
    model = SAC(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=0.0003,
        buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef='auto',  # 自动调整熵系数
    )

    print(f"\n训练配置:")
    print(f"  算法: SAC (Soft Actor-Critic)")
    print(f"  总训练步数: {total_timesteps:,}")
    print(f"  训练数据点: {len(df)}")
    print(f"  学习率: 0.0003")
    print(f"  批次大小: 256")
    print(f"  风险管理: 止损10%, 止盈20%")
    print(f"  奖励机制: 收益 + 风险控制 + 卖出鼓励")

    print("\n开始训练...")
    model.learn(total_timesteps=total_timesteps)

    model_path = f"sac_{ticker.lower()}_advanced"
    model.save(model_path)
    print(f"\n✅ 模型已保存: {model_path}.zip")

    return model

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print("🚀 高级版 AAPL 股票交易 AI 训练系统")
    print("=" * 70)

    TICKER = 'AAPL'
    START_DATE = '2015-01-01'
    END_DATE = '2024-01-01'
    TRAIN_TEST_SPLIT = 0.8
    TOTAL_TIMESTEPS = 100000

    print(f"目标股票: {TICKER}")
    print(f"数据范围: {START_DATE} - {END_DATE}")
    print(f"训练步数: {TOTAL_TIMESTEPS:,}")
    print(f"算法: SAC (Soft Actor-Critic)")
    print("=" * 70)

    # 1. 下载数据
    df = download_and_prepare_data(TICKER, START_DATE, END_DATE)
    if df is None:
        exit(1)

    # 2. 添加技术指标
    df = add_technical_indicators(df)

    # 3. 分割数据
    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split_idx].copy()

    print(f"\n数据分割:")
    print(f"  训练集: {len(train_df)} 天")

    # 4. 训练模型
    model = train_advanced_model(train_df, TICKER, total_timesteps=TOTAL_TIMESTEPS)

    print("\n✅ 训练完成!")
    print(f"模型文件: sac_{TICKER.lower()}_advanced.zip")
    print("\n高级特性:")
    print("  ✅ SAC 算法 (更强探索能力)")
    print("  ✅ 止损/止盈机制")
    print("  ✅ 风险管理 (回撤控制)")
    print("  ✅ 鼓励卖出 (防止只买不卖)")
