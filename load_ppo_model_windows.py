"""
如何加载和使用已训练的 PPO 模型 (Windows 版本)
=====================================
这个脚本展示如何:
1. 加载保存的 PPO 模型
2. 在新数据上测试
3. 可视化交易决策 (支持中文显示)
"""

import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

# ==========================================
# 配置 matplotlib 支持中文 (Windows)
# ==========================================
import matplotlib
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("配置中文字体...")
print("=" * 60)

# Windows 常见中文字体
chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']

# 查找可用的中文字体
available_fonts = [f.name for f in fm.fontManager.ttflist]
selected_font = None

for font in chinese_fonts:
    if font in available_fonts:
        selected_font = font
        print(f"✅ 找到中文字体: {font}")
        break

if selected_font:
    matplotlib.rcParams['font.sans-serif'] = [selected_font]
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    print(f"✅ 已设置中文字体: {selected_font}\n")
else:
    print("⚠️ 未找到中文字体，将使用英文标题")
    print("安装中文字体方法: 系统 > 设置 > 字体\n")
    # 如果没有中文字体，使用英文标题
    USE_ENGLISH = True

# ==========================================
# 1. 定义交易环境
# ==========================================
class SimpleTradingEnv(gym.Env):
    """简化的交易环境"""

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
# 2. 生成测试数据
# ==========================================
def generate_test_data(n_days=200):
    np.random.seed(123)
    price = 120
    prices = []
    for _ in range(n_days):
        change = np.random.randn() * 2.5
        price = max(price + change, 10)
        prices.append(price)

    df = pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_days)
    })

    df['sma_10'] = df['close'].rolling(10).mean().fillna(df['close'])
    df['sma_30'] = df['close'].rolling(30).mean().fillna(df['close'])

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)

    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd'] = df['macd'].fillna(0)

    return df

# ==========================================
# 3. 加载模型并测试
# ==========================================
def load_and_test_model(model_path, df):
    print("=" * 60)
    print("加载 PPO 交易模型")
    print("=" * 60)

    try:
        model = PPO.load(model_path)
        print(f"✅ 模型加载成功: {model_path}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

    env = SimpleTradingEnv(df)
    print(f"✅ 测试环境已创建 (数据点: {len(df)})")

    print("\n开始测试...")
    obs, _ = env.reset()
    done = False

    actions_history = []
    prices_history = []
    balance_history = []
    shares_history = []
    total_asset_history = []

    step = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        actions_history.append(action)
        prices_history.append(env.df.iloc[env.current_step - 1]['close'])
        balance_history.append(env.balance)
        shares_history.append(env.shares_held)
        total_asset = env.balance + env.shares_held * env.df.iloc[env.current_step - 1]['close']
        total_asset_history.append(total_asset)
        step += 1

    final_profit = env.total_profit
    final_return = (final_profit / env.initial_balance) * 100
    initial_value = env.initial_balance
    final_value = env.balance + env.shares_held * env.df.iloc[-1]['close']

    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"初始资金:     ${initial_value:,.2f}")
    print(f"最终资产:     ${final_value:,.2f}")
    print(f"总收益:       ${final_profit:,.2f}")
    print(f"收益率:       {final_return:.2f}%")
    print(f"交易次数:     {step} 步")

    actions_count = {
        0: actions_history.count(0),
        1: actions_history.count(1),
        2: actions_history.count(2)
    }
    print(f"\n动作分布:")
    print(f"  卖出 (0): {actions_count[0]} 次 ({actions_count[0]/len(actions_history)*100:.1f}%)")
    print(f"  持有 (1): {actions_count[1]} 次 ({actions_count[1]/len(actions_history)*100:.1f}%)")
    print(f"  买入 (2): {actions_count[2]} 次 ({actions_count[2]/len(actions_history)*100:.1f}%)")

    visualize_results(
        prices_history,
        actions_history,
        total_asset_history,
        balance_history,
        shares_history,
        use_chinese=selected_font is not None
    )

    return {
        'profit': final_profit,
        'return': final_return,
        'actions': actions_history,
        'prices': prices_history,
        'total_assets': total_asset_history
    }

# ==========================================
# 4. 可视化 (支持中文/英文)
# ==========================================
def visualize_results(prices, actions, total_assets, balance, shares, use_chinese=True):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # 根据是否支持中文选择标题
    if use_chinese:
        titles = ['价格走势和交易决策', '总资产变化', '现金余额', '持股数量']
        labels = {
            'price': '股价', 'buy': '买入', 'sell': '卖出',
            'asset': '总资产', 'initial': '初始资金',
            'cash': '现金余额', 'shares': '持股数量',
            'time': '时间步', 'value': '价格 ($)'
        }
    else:
        titles = ['Price & Trading Decisions', 'Total Asset', 'Cash Balance', 'Shares Held']
        labels = {
            'price': 'Stock Price', 'buy': 'Buy', 'sell': 'Sell',
            'asset': 'Total Asset', 'initial': 'Initial Capital',
            'cash': 'Cash Balance', 'shares': 'Shares',
            'time': 'Time Step', 'value': 'Price ($)'
        }

    # 1. 价格和买卖点
    ax1 = axes[0]
    ax1.plot(prices, label=labels['price'], color='black', linewidth=1.5)

    buy_points = [i for i, a in enumerate(actions) if a == 2]
    sell_points = [i for i, a in enumerate(actions) if a == 0]

    if buy_points:
        ax1.scatter(buy_points, [prices[i] for i in buy_points],
                   color='green', marker='^', s=100, label=labels['buy'], zorder=5)
    if sell_points:
        ax1.scatter(sell_points, [prices[i] for i in sell_points],
                   color='red', marker='v', s=100, label=labels['sell'], zorder=5)

    ax1.set_title(titles[0], fontsize=14, fontweight='bold')
    ax1.set_ylabel(labels['value'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 总资产
    ax2 = axes[1]
    ax2.plot(total_assets, label=labels['asset'], color='blue', linewidth=2)
    ax2.axhline(y=10000, color='gray', linestyle='--', label=labels['initial'])
    ax2.set_title(titles[1], fontsize=14, fontweight='bold')
    ax2.set_ylabel('Total Asset ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 现金
    ax3 = axes[2]
    ax3.plot(balance, label=labels['cash'], color='green', linewidth=1.5)
    ax3.set_title(titles[2], fontsize=14, fontweight='bold')
    ax3.set_ylabel('Cash ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 持股
    ax4 = axes[3]
    ax4.plot(shares, label=labels['shares'], color='orange', linewidth=1.5)
    ax4.set_title(titles[3], fontsize=14, fontweight='bold')
    ax4.set_xlabel(labels['time'])
    ax4.set_ylabel('Shares')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ppo_trading_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 图表已保存: ppo_trading_analysis.png")
    plt.show()

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    model_path = "ppo_trading_model.zip"

    print("生成测试数据...")
    test_df = generate_test_data(n_days=200)

    results = load_and_test_model(model_path, test_df)

    if results:
        print("\n✅ 测试完成!")
        print(f"最终收益率: {results['return']:.2f}%")
