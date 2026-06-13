"""
测试高级版 SAC AAPL 模型
========================
测试包含所有高级特性的 SAC 模型
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')

# 配置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==========================================
# 高级交易环境 (必须与训练时一致)
# ==========================================
class AdvancedTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000,
                 stop_loss_pct=0.1, take_profit_pct=0.2):
        super(AdvancedTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0
        self.total_trades = 0
        self.buy_price = 0
        self.max_portfolio_value = self.initial_balance
        self.consecutive_losses = 0
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        current_price = float(row['close'])
        total_value = self.balance + self.shares_held * current_price

        stock_ratio = (self.shares_held * current_price) / total_value if total_value > 0 else 0
        cash_ratio = self.balance / total_value if total_value > 0 else 1
        unrealized_pnl = (current_price - self.buy_price) / self.buy_price if self.buy_price > 0 else 0
        drawdown = (self.max_portfolio_value - total_value) / self.max_portfolio_value if self.max_portfolio_value > 0 else 0

        sma_10 = float(row.get('sma_10', current_price))
        sma_30 = float(row.get('sma_30', current_price))
        price_to_sma10 = (current_price - sma_10) / sma_10 if sma_10 > 0 else 0
        price_to_sma30 = (current_price - sma_30) / sma_30 if sma_30 > 0 else 0

        obs = np.array([
            float(self.shares_held),
            float(self.balance),
            current_price,
            sma_10,
            sma_30,
            float(row.get('sma_50', current_price)),
            float(row.get('rsi', 50)),
            float(row.get('macd', 0)),
            float(row.get('macd_signal', 0)),
            float(row.get('bb_upper', current_price)),
            float(row.get('bb_lower', current_price)),
            float(row.get('volume', 0)),
            float(self.total_profit),
            float(stock_ratio),
            float(cash_ratio),
            float(unrealized_pnl),
            float(drawdown),
            float(price_to_sma10),
        ], dtype=np.float32)
        return obs

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = float(action[0])
        else:
            action = float(action)

        action = np.clip(action, -1.0, 1.0)
        current_price = float(self.df.iloc[self.current_step]['close'])

        # 自动止损/止盈
        auto_sell = False
        if self.shares_held > 0 and self.buy_price > 0:
            current_return = (current_price - self.buy_price) / self.buy_price
            if current_return < -self.stop_loss_pct:
                action = -1.0
                auto_sell = True
            elif current_return > self.take_profit_pct:
                action = -0.5
                auto_sell = True

        # 执行交易
        if action < -0.1:
            sell_ratio = abs(action)
            shares_to_sell = int(self.shares_held * sell_ratio)
            if shares_to_sell > 0:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                self.total_trades += 1
                if self.shares_held == 0:
                    self.buy_price = 0

        elif action > 0.1:
            buy_ratio = action
            max_can_buy = int(self.balance // current_price)
            shares_to_buy = int(max_can_buy * buy_ratio)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                if self.shares_held > 0 and self.buy_price > 0:
                    total_cost = self.shares_held * self.buy_price + cost
                    self.shares_held += shares_to_buy
                    self.buy_price = total_cost / self.shares_held
                else:
                    self.buy_price = current_price
                    self.shares_held = shares_to_buy
                self.balance -= cost
                self.total_trades += 1

        new_total_value = self.balance + self.shares_held * current_price
        self.total_profit = new_total_value - self.initial_balance

        if new_total_value > self.max_portfolio_value:
            self.max_portfolio_value = new_total_value

        reward = self.total_profit / self.initial_balance

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self._get_observation()

        return obs, float(reward), done, False, {'auto_sell': auto_sell}

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

    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

print("=" * 70)
print("📈 测试高级版 SAC AAPL 交易模型")
print("=" * 70)

# 1. 加载模型
model_path = "C:/Users/Silvi/sac_aapl_advanced.zip"
print(f"\n加载高级版 SAC 模型: {model_path}")

try:
    model = SAC.load(model_path)
    print("✅ 模型加载成功!")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit(1)

# 2. 下载2024年测试数据
print("\n下载 2024 年 AAPL 测试数据...")
try:
    import yfinance as yf
    test_df = yf.download('AAPL', start='2024-01-01', end='2024-12-18', progress=False)

    test_df = test_df.rename(columns={
        'Close': 'close', 'Volume': 'volume',
        'Open': 'open', 'High': 'high', 'Low': 'low'
    })
    test_df = test_df.reset_index()
    print(f"✅ 下载了 {len(test_df)} 天的数据")

except Exception as e:
    print(f"❌ 下载失败: {e}")
    exit(1)

# 3. 添加技术指标
test_df = add_technical_indicators(test_df)

# 4. 运行测试
print("\n" + "=" * 70)
print("开始测试...")
print("=" * 70)

env = AdvancedTradingEnv(test_df)
obs, _ = env.reset()
done = False

actions_history = []
prices_history = []
total_assets_history = []
auto_sells = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    if info.get('auto_sell', False):
        auto_sells += 1

    actions_history.append(float(action[0]) if isinstance(action, np.ndarray) else float(action))
    prices_history.append(float(env.df.iloc[env.current_step - 1]['close']))
    total_asset = env.balance + env.shares_held * float(env.df.iloc[env.current_step - 1]['close'])
    total_assets_history.append(float(total_asset))

# 5. 计算结果
final_profit = float(env.total_profit)
final_return = (final_profit / env.initial_balance) * 100
buy_hold_return = ((float(prices_history[-1]) - float(prices_history[0])) / float(prices_history[0])) * 100

print("\n" + "=" * 70)
print("📊 高级版 SAC 模型测试结果")
print("=" * 70)

print(f"\n💰 财务表现:")
print(f"  初始资金:        ${env.initial_balance:,.2f}")
print(f"  最终资产:        ${env.initial_balance + final_profit:,.2f}")
print(f"  总收益:          ${final_profit:,.2f}")

print(f"\n📈 收益率对比:")
print(f"  SAC 高级策略:    {final_return:+.2f}%")
print(f"  买入持有策略:    {buy_hold_return:+.2f}%")
print(f"  超额收益:        {final_return - buy_hold_return:+.2f}%")

print(f"\n📊 交易统计:")
print(f"  总交易次数:      {env.total_trades}")
print(f"  自动止损/止盈:   {auto_sells} 次")
print(f"  平均每日交易:    {env.total_trades / len(test_df):.2f} 次")

# 分析动作
buy_actions = sum(1 for a in actions_history if a > 0.1)
sell_actions = sum(1 for a in actions_history if a < -0.1)
hold_actions = sum(1 for a in actions_history if -0.1 <= a <= 0.1)
total = len(actions_history)

print(f"\n🎯 决策分布:")
print(f"  买入决策: {buy_actions} 次 ({buy_actions/total*100:.1f}%)")
print(f"  持有决策: {hold_actions} 次 ({hold_actions/total*100:.1f}%)")
print(f"  卖出决策: {sell_actions} 次 ({sell_actions/total*100:.1f}%)")

# 6. 可视化对比
fig, axes = plt.subplots(3, 1, figsize=(16, 14))

# 价格和交易
ax1 = axes[0]
ax1.plot(prices_history, label='AAPL 股价', color='black', linewidth=2)

buy_points = [i for i, a in enumerate(actions_history) if a > 0.1]
sell_points = [i for i, a in enumerate(actions_history) if a < -0.1]

if buy_points:
    ax1.scatter(buy_points, [prices_history[i] for i in buy_points],
               color='green', marker='^', s=100, label=f'买入 ({len(buy_points)}次)', zorder=5, alpha=0.7)
if sell_points:
    ax1.scatter(sell_points, [prices_history[i] for i in sell_points],
               color='red', marker='v', s=100, label=f'卖出 ({len(sell_points)}次)', zorder=5, alpha=0.7)

ax1.set_title('SAC 高级版: AAPL 价格和交易决策 (2024年)', fontsize=16, fontweight='bold')
ax1.set_ylabel('股价 ($)', fontsize=12)
ax1.legend(fontsize=11, loc='best')
ax1.grid(True, alpha=0.3)

# 资产对比
ax2 = axes[1]
ax2.plot(total_assets_history, label=f'SAC 高级策略 ({final_return:+.2f}%)',
         color='purple', linewidth=2.5)
ax2.axhline(y=10000, color='gray', linestyle='--', label='初始资金', linewidth=1.5)

buy_hold_assets = [10000 * (1 + (p - prices_history[0]) / prices_history[0]) for p in prices_history]
ax2.plot(buy_hold_assets, label=f'买入持有 ({buy_hold_return:+.2f}%)',
         color='orange', linewidth=2, alpha=0.7, linestyle='--')

ax2.set_title('投资组合价值对比', fontsize=16, fontweight='bold')
ax2.set_ylabel('总资产 ($)', fontsize=12)
ax2.legend(fontsize=11, loc='best')
ax2.grid(True, alpha=0.3)

# 动作强度
ax3 = axes[2]
action_colors = ['red' if a < -0.1 else 'green' if a > 0.1 else 'gray' for a in actions_history]
ax3.bar(range(len(actions_history)), actions_history, color=action_colors, alpha=0.6)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.axhline(y=0.1, color='green', linestyle='--', linewidth=0.5, alpha=0.3)
ax3.axhline(y=-0.1, color='red', linestyle='--', linewidth=0.5, alpha=0.3)
ax3.set_title('SAC 动作强度 (连续动作空间)', fontsize=16, fontweight='bold')
ax3.set_xlabel('交易日', fontsize=12)
ax3.set_ylabel('动作值 [-1.0 ~ 1.0]', fontsize=12)
ax3.set_ylim(-1.1, 1.1)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('aapl_sac_advanced_results.png', dpi=150, bbox_inches='tight')
print(f"\n📊 图表已保存: aapl_sac_advanced_results.png")
plt.show()

print("\n" + "=" * 70)
print("✅ 测试完成!")
print("=" * 70)

# 性能评估
if final_return > buy_hold_return + 1:
    print(f"\n🎉🎉🎉 SAC 策略跑赢大盘 {final_return - buy_hold_return:.2f}%!")
    print("风险管理和智能卖出机制起作用了!")
elif final_return > buy_hold_return:
    print(f"\n✅ SAC 策略略微跑赢大盘 {final_return - buy_hold_return:.2f}%")
elif final_return > 0:
    print(f"\n✅ SAC 策略盈利,但未跑赢大盘")
else:
    print(f"\n⚠️ SAC 策略出现亏损")

print(f"\n🔥 高级特性:")
print(f"  ✅ 使用 SAC 算法")
print(f"  ✅ 止损/止盈: {auto_sells} 次自动触发")
print(f"  ✅ 卖出决策: {sell_actions} 次 ({sell_actions/total*100:.1f}%)")
print(f"  ✅ 风险管理和回撤控制")
