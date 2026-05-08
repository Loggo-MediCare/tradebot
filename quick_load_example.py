"""
快速加载 PPO 模型示例 - 最简版本
===================================
只需 5 行代码即可加载和使用模型!
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from stable_baselines3 import PPO
import numpy as np

# ==========================================
# 方法 1: 最简单的加载方式
# ==========================================
print("=" * 60)
print("方法 1: 基本加载")
print("=" * 60)

# 加载模型
model = PPO.load("ppo_trading_model")
print("✅ 模型加载成功!")

# 查看模型信息
print(f"\n模型类型: {type(model)}")
print(f"策略网络: {model.policy}")
print(f"学习率: {model.learning_rate}")
print(f"折扣因子 (gamma): {model.gamma}")

# ==========================================
# 方法 2: 加载并查看模型参数
# ==========================================
print("\n" + "=" * 60)
print("方法 2: 查看模型详细信息")
print("=" * 60)

# 获取模型的所有参数
params = model.get_parameters()
print(f"\n模型参数键: {list(params.keys())[:5]}...")  # 显示前5个键

# ==========================================
# 方法 3: 使用模型进行预测
# ==========================================
print("\n" + "=" * 60)
print("方法 3: 使用模型预测动作")
print("=" * 60)

# 创建一个假的观察状态 (必须是 shape=(10,) 的数组)
fake_observation = np.array([
    0,          # 持有股票数量
    10000,      # 账户余额
    100,        # 当前价格
    95,         # SMA 10
    92,         # SMA 30
    50,         # RSI
    2,          # MACD
    5000,       # 成交量
    0,          # 总收益
    0.5         # 进度
], dtype=np.float32)

# 预测动作
action, _states = model.predict(fake_observation, deterministic=True)

action_names = {0: "卖出", 1: "持有", 2: "买入"}
print(f"\n给定状态:")
print(f"  - 持有股票: {fake_observation[0]}")
print(f"  - 账户余额: ${fake_observation[1]:,.2f}")
print(f"  - 当前价格: ${fake_observation[2]:,.2f}")
print(f"\n模型预测动作: {action} ({action_names[int(action)]})")

# ==========================================
# 方法 4: 保存模型到不同位置
# ==========================================
print("\n" + "=" * 60)
print("方法 4: 重新保存模型")
print("=" * 60)

# 保存到新位置
new_path = "backup_ppo_trading_model"
model.save(new_path)
print(f"✅ 模型已备份至: {new_path}.zip")

# ==========================================
# 方法 5: 从不同路径加载
# ==========================================
print("\n" + "=" * 60)
print("方法 5: 从不同路径加载")
print("=" * 60)

# 可以指定完整路径
import os

# 当前目录
current_dir = os.getcwd()
print(f"当前目录: {current_dir}")

# 列出当前目录的 .zip 文件
zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]
print(f"\n找到的模型文件:")
for f in zip_files:
    print(f"  - {f}")

# ==========================================
# 总结
# ==========================================
print("\n" + "=" * 60)
print("总结: 如何使用 PPO 模型")
print("=" * 60)

print("""
1️⃣ 基本加载:
   from stable_baselines3 import PPO
   model = PPO.load("ppo_trading_model")

2️⃣ 预测动作:
   action, _ = model.predict(observation, deterministic=True)

3️⃣ 保存模型:
   model.save("new_model_path")

4️⃣ 动作含义:
   0 = 卖出
   1 = 持有
   2 = 买入

5️⃣ 观察空间 (必须是 10 个元素的 numpy 数组):
   [持股数, 余额, 价格, SMA10, SMA30, RSI, MACD, 成交量, 收益, 进度]
""")

print("\n✅ 完成!")
