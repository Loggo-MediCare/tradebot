"""
如何加载和使用已训练的 PPO 模型
=====================================
这个脚本展示如何:
1. 加载保存的 PPO 模型
2. 在新数据上测试
3. 可视化交易决策
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

print("【步骤1】正在安装中文字体...")
print("=" * 60)

# 更新包管理器
subprocess.run(['apt-get', 'update'],
               stdout=subprocess.DEVNULL,
               stderr=subprocess.DEVNULL,
               check=False
)

# 安装开源中文字体（文泉驿微米黑，更常用且兼容性好）
subprocess.run(['apt-get', 'install', '-y', 'fonts-wqy-microhei'],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=False
)
print("✓ 字体安装完成：fonts-wqy-microhei")

# 配置 matplotlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore') # 忽略可能的字体警告

print("\n【步骤2】正在配置 matplotlib...")
print("-" * 60)

# 清除 matplotlib 字体缓存，以确保新安装的字体被识别
cache_dir = os.path.expanduser('~/.matplotlib')
for cache_file in [
    os.path.join(cache_dir, 'fontList.json'),
    os.path.join(cache_dir, 'fontList.cache')
]:
    if os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            print(f"✓ 已清除旧字体缓存：{os.path.basename(cache_file)}")
        except Exception as e:
            print(f"⚠️ 清除字体缓存失败：{e}")

# 强制添加字体文件到matplotlib
# 注意：这里我们添加 wqy-microhei.ttc，其对应的matplotlib名称通常是 'WenQuanYi Micro Hei'
font_path_wqy = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
if os.path.exists(font_path_wqy):
    try:
        fm.fontManager.addfont(font_path_wqy)
        print(f"✓ 已手动添加字体文件：{os.path.basename(font_path_wqy)}")
    except Exception as e:
        print(f"⚠️ 添加字体文件失败：{e}")

# 设置中文字体（优先级顺序）
# 'WenQuanYi Micro Hei' 是 'wqy-microhei.ttc' 在matplotlib中的名称
font_options = [
    'WenQuanYi Micro Hei',  # 文泉驿微米黑（优先级最高）
    'Noto Sans CJK SC',     # Google Noto 字体
    'SimHei',               # 微软黑体（如果有的话）
    'DejaVu Sans'           # 备选方案
]

selected_font = None
for font in font_options:
    # 检查字体是否可用，避免设置一个不存在的字体导致警告
    if font in [f.name for f in fm.fontManager.ttflist]:
        plt.rcParams['font.sans-serif'] = [font]
        selected_font = font
        break

if not selected_font:
    # 如果上述字体都不可用，则回退到默认sans-serif（可能不显示中文）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    print("⚠️ 未能找到合适的中文字体，已设置为'DejaVu Sans'，中文可能无法正常显示。")
else:
    print(f"✓ 已设置中文字体：{selected_font}")

# 防止负号显示为方框
plt.rcParams['axes.unicode_minus'] = False

# Colab 特定配置
plt.rcParams['figure.max_open_warning'] = 50
plt.rcParams['font.size'] = 10

# 重建字体缓存 (在较新版本Matplotlib中通常不需要显式调用，但删除文件后会自动重建)
# fm.fontManager.rebuild() # 此行已移除，因其在最新版matplotlib中已被移除

print("\n✅ 中文字体配置完成！")
print("现在可以正常显示中文图表了\n")


# ==========================================
# 1. 定义相同的交易环境 (必须与训练时一致!)
# ==========================================
class SimpleTradingEnv(gym.Env):
    """简化的交易环境 - 必须与训练时完全相同"""

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
            self.shares_held,
            self.balance,
            row['close'],
            row.get('sma_10', 0),
            row.get('sma_30', 0),
            row.get('rsi', 50),
            row.get('macd', 0),
            row.get('volume', 0),
            self.total_profit,
            self.current_step / len(self.df)
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

        # 计算总资产
        total_asset = self.balance + self.shares_held * current_price
        self.total_profit = total_asset - self.initial_balance
        reward = self.total_profit / self.initial_balance

        # 下一步
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = False
        obs = self._get_observation()

        return obs, reward, done, truncated, {}

# ==========================================
# 2. 生成测试数据
# ==========================================
def generate_test_data(n_days=200):
    """生成测试数据"""
    np.random.seed(123)  # 不同的种子,模拟新数据

    price = 120  # 不同的起始价格
    prices = []
    for _ in range(n_days):
        change = np.random.randn() * 2.5
        price = max(price + change, 10)
        prices.append(price)

    df = pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_days)
    })

    # 添加技术指标 (必须与训练时一致)
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
# 3. 加载模型并测试
# ==========================================
def load_and_test_model(model_path, df):
    """
    加载模型并测试

    参数:
        model_path: 模型文件路径 (例如: "ppo_trading_model.zip")
        df: 测试数据 DataFrame
    """
    print("=" * 60)
    print("加载 PPO 交易模型")
    print("=" * 60)

    # 1. 加载模型
    try:
        model = PPO.load(model_path)
        print(f"✅ 模型加载成功: {model_path}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

    # 2. 创建测试环境
    env = SimpleTradingEnv(df)
    print(f"✅ 测试环境已创建 (数据点: {len(df)})")

    # 3. 运行测试
    print("\n开始测试...")
    obs, _ = env.reset()
    done = False

    # 记录交易历史
    actions_history = []
    prices_history = []
    balance_history = []
    shares_history = []
    total_asset_history = []

    step = 0
    while not done:
        # 使用模型预测动作 (deterministic=True 表示不随机探索)
        action, _states = model.predict(obs, deterministic=True)

        # 执行动作
        obs, reward, done, truncated, info = env.step(action)

        # 记录
        actions_history.append(action)
        prices_history.append(env.df.iloc[env.current_step - 1]['close'])
        balance_history.append(env.balance)
        shares_history.append(env.shares_held)
        total_asset = env.balance + env.shares_held * env.df.iloc[env.current_step - 1]['close']
        total_asset_history.append(total_asset)

        step += 1

    # 4. 计算结果
    final_profit = env.total_profit
    final_return = (final_profit / env.initial_balance) * 100
    initial_value = env.initial_balance
    final_value = env.balance + env.shares_held * env.df.iloc[-1]['close']

    # 5. 打印结果
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"初始资金:     ${initial_value:,.2f}")
    print(f"最终资产:     ${final_value:,.2f}")
    print(f"总收益:       ${final_profit:,.2f}")
    print(f"收益率:       {final_return:.2f}%")
    print(f"交易次数:     {step} 步")

    # 统计动作分布
    actions_count = {
        0: actions_history.count(0),  # 卖出
        1: actions_history.count(1),  # 持有
        2: actions_history.count(2)   # 买入
    }
    print(f"\n动作分布:")
    print(f"  卖出 (0): {actions_count[0]} 次 ({actions_count[0]/len(actions_history)*100:.1f}%)")
    print(f"  持有 (1): {actions_count[1]} 次 ({actions_count[1]/len(actions_history)*100:.1f}%)")
    print(f"  买入 (2): {actions_count[2]} 次 ({actions_count[2]/len(actions_history)*100:.1f}%)")

    # 6. 可视化
    visualize_results(
        prices_history,
        actions_history,
        total_asset_history,
        balance_history,
        shares_history
    )

    return {
        'profit': final_profit,
        'return': final_return,
        'actions': actions_history,
        'prices': prices_history,
        'total_assets': total_asset_history
    }

# ==========================================
# 4. 可视化交易结果
# ==========================================
def visualize_results(prices, actions, total_assets, balance, shares):
    """可视化交易结果"""

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # 1. 价格和买卖点
    ax1 = axes[0]
    ax1.plot(prices, label='股价', color='black', linewidth=1.5)

    # 标记买入和卖出点
    buy_points = [i for i, a in enumerate(actions) if a == 2]
    sell_points = [i for i, a in enumerate(actions) if a == 0]

    if buy_points:
        ax1.scatter(buy_points, [prices[i] for i in buy_points],
                   color='green', marker='^', s=100, label='买入', zorder=5)
    if sell_points:
        ax1.scatter(sell_points, [prices[i] for i in sell_points],
                   color='red', marker='v', s=100, label='卖出', zorder=5)

    ax1.set_title('价格走势和交易决策', fontsize=14, fontweight='bold')
    ax1.set_ylabel('价格 ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 总资产变化
    ax2 = axes[1]
    ax2.plot(total_assets, label='总资产', color='blue', linewidth=2)
    ax2.axhline(y=10000, color='gray', linestyle='--', label='初始资金')
    ax2.set_title('总资产变化', fontsize=14, fontweight='bold')
    ax2.set_ylabel('总资产 ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 现金余额
    ax3 = axes[2]
    ax3.plot(balance, label='现金余额', color='green', linewidth=1.5)
    ax3.set_title('现金余额', fontsize=14, fontweight='bold')
    ax3.set_ylabel('现金 ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 持股数量
    ax4 = axes[3]
    ax4.plot(shares, label='持股数量', color='orange', linewidth=1.5)
    ax4.set_title('持股数量', fontsize=14, fontweight='bold')
    ax4.set_xlabel('时间步')
    ax4.set_ylabel('股票数量')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ppo_trading_analysis.png', dpi=150)
    print(f"\n📊 图表已保存: ppo_trading_analysis.png")
    plt.show()

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 模型路径
    model_path = "ppo_trading_model.zip"

    # 2. 生成测试数据
    print("生成测试数据...")
    test_df = generate_test_data(n_days=200)

    # 3. 加载模型并测试
    results = load_and_test_model(model_path, test_df)

    if results:
        print("\n✅ 测试完成!")
        print(f"最终收益率: {results['return']:.2f}%")
