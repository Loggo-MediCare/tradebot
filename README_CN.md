# PPO 交易模型使用指南

## 📁 文件说明

### 已保存的模型文件:
- `ppo_trading_model.zip` - 训练好的 PPO 模型 ✅
- `backup_ppo_trading_model.zip` - 备份模型

### 代码文件:
1. **simple_finrl_example.py** - 完整的训练脚本
2. **load_ppo_model.py** - 详细的模型加载和测试脚本 ⭐推荐
3. **quick_load_example.py** - 快速示例(5分钟了解)

---

## 🚀 快速开始

### 方法 1: 最简单的方式 (3行代码)

```python
from stable_baselines3 import PPO

# 加载模型
model = PPO.load("ppo_trading_model")

# 使用模型预测
observation = [0, 10000, 100, 95, 92, 50, 2, 5000, 0, 0.5]
action, _ = model.predict(observation, deterministic=True)

# action 的含义:
# 0 = 卖出
# 1 = 持有
# 2 = 买入
```

---

## 📊 完整测试流程

### 运行完整测试:

```bash
python load_ppo_model.py
```

这将:
1. ✅ 加载训练好的模型
2. ✅ 生成测试数据
3. ✅ 运行完整的交易模拟
4. ✅ 计算收益率
5. ✅ 生成可视化图表 (`ppo_trading_analysis.png`)

### 预期输出:

```
============================================================
测试结果
============================================================
初始资金:     $10,000.00
最终资产:     $10,XXX.XX
总收益:       $XXX.XX
收益率:       X.XX%
交易次数:     200 步

动作分布:
  卖出 (0): XX 次 (XX.X%)
  持有 (1): XX 次 (XX.X%)
  买入 (2): XX 次 (XX.X%)
```

---

## 🔍 模型详细信息

### 模型架构:

从运行结果可以看到:

```python
ActorCriticPolicy(
  策略网络 (Policy Net):
    - 输入层: 10 个特征
    - 隐藏层 1: 64 神经元 (Tanh 激活)
    - 隐藏层 2: 64 神经元 (Tanh 激活)
    - 输出层: 3 个动作

  价值网络 (Value Net):
    - 输入层: 10 个特征
    - 隐藏层 1: 64 神经元 (Tanh 激活)
    - 隐藏层 2: 64 神经元 (Tanh 激活)
    - 输出层: 1 个值(状态价值)
)
```

### 训练参数:

- **算法**: PPO (Proximal Policy Optimization)
- **学习率**: 0.0003
- **折扣因子 (Gamma)**: 0.99
- **批次大小**: 64
- **训练步数**: 20,000

---

## 📋 观察空间说明

模型需要 **10 个特征** 作为输入:

| 索引 | 特征名 | 说明 | 示例值 |
|------|--------|------|--------|
| 0 | shares_held | 当前持有股票数量 | 0 |
| 1 | balance | 账户余额 | 10000 |
| 2 | close | 当前股价 | 100 |
| 3 | sma_10 | 10日简单移动平均 | 95 |
| 4 | sma_30 | 30日简单移动平均 | 92 |
| 5 | rsi | 相对强弱指标 (14日) | 50 |
| 6 | macd | MACD 指标 | 2 |
| 7 | volume | 成交量 | 5000 |
| 8 | total_profit | 累计收益 | 0 |
| 9 | progress | 交易进度 (0-1) | 0.5 |

### 使用示例:

```python
import numpy as np

# 创建观察数组
observation = np.array([
    0,          # 0股持仓
    10000,      # $10,000 余额
    120.5,      # 当前价格 $120.5
    118.2,      # SMA 10
    115.8,      # SMA 30
    65.3,       # RSI
    1.5,        # MACD
    8500,       # 成交量
    250,        # 已盈利 $250
    0.3         # 完成了 30%
], dtype=np.float32)

# 预测
action, _ = model.predict(observation, deterministic=True)
```

---

## 🎯 动作空间

模型输出 **离散动作** (0, 1, 或 2):

| 动作值 | 含义 | 描述 |
|--------|------|------|
| 0 | 卖出 (SELL) | 卖出所有持仓 |
| 1 | 持有 (HOLD) | 不做任何操作 |
| 2 | 买入 (BUY) | 用全部余额买入 |

---

## 🔄 在真实数据上使用模型

### 步骤 1: 准备您的数据

```python
import pandas as pd

# 加载真实股价数据
df = pd.read_csv('your_stock_data.csv')

# 必须包含这些列:
# - close: 收盘价
# - volume: 成交量

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
```

### 步骤 2: 使用模型

```python
from stable_baselines3 import PPO

# 加载模型
model = PPO.load("ppo_trading_model")

# 遍历每一天
for i in range(len(df)):
    # 构建观察
    obs = np.array([
        shares_held,
        balance,
        df.iloc[i]['close'],
        df.iloc[i]['sma_10'],
        df.iloc[i]['sma_30'],
        df.iloc[i]['rsi'],
        df.iloc[i]['macd'],
        df.iloc[i]['volume'],
        total_profit,
        i / len(df)
    ], dtype=np.float32)

    # 预测动作
    action, _ = model.predict(obs, deterministic=True)

    # 执行动作
    if action == 0:  # 卖出
        # 您的卖出逻辑
        pass
    elif action == 2:  # 买入
        # 您的买入逻辑
        pass
```

---

## 📈 可视化分析

运行 `load_ppo_model.py` 后会生成图表:

1. **价格走势和交易决策** - 显示买入/卖出点
2. **总资产变化** - 追踪投资组合价值
3. **现金余额** - 可用现金变化
4. **持股数量** - 持仓变化

图表保存为: `ppo_trading_analysis.png`

---

## ⚙️ 重新训练模型

如果您想用新数据重新训练:

```bash
python simple_finrl_example.py
```

修改参数:
```python
# 在 simple_finrl_example.py 中修改:
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=0.0003,     # 学习率
    n_steps=2048,             # 每次更新的步数
    batch_size=64,            # 批次大小
    n_epochs=10,              # 每次更新的训练轮数
    gamma=0.99,               # 折扣因子
)

model.learn(total_timesteps=50000)  # 训练步数
```

---

## 🐛 常见问题

### Q1: 模型文件在哪里?
**A**: 在当前工作目录下,文件名为 `ppo_trading_model.zip`

### Q2: 如何改变模型路径?
**A**:
```python
# 保存到指定位置
model.save("C:/path/to/your/model")

# 加载
model = PPO.load("C:/path/to/your/model")
```

### Q3: 观察空间维度不匹配?
**A**: 确保观察数组有 **exactly 10 个元素**,并且是 `float32` 类型

### Q4: 模型预测总是相同?
**A**: 使用 `deterministic=True` 会得到一致的预测。如果想要探索性预测,使用 `deterministic=False`

### Q5: 可以用于实盘交易吗?
**A**: ⚠️ **不建议直接用于实盘!** 这个模型:
- 在模拟数据上训练
- 未考虑交易成本、滑点
- 未经过严格的回测验证

建议:
1. 用真实历史数据重新训练
2. 进行充分的回测
3. 从小资金开始纸面交易
4. 逐步验证策略有效性

---

## 📚 相关资源

- **Stable Baselines3 文档**: https://stable-baselines3.readthedocs.io/
- **PPO 算法论文**: https://arxiv.org/abs/1707.06347
- **FinRL 项目**: https://github.com/AI4Finance-Foundation/FinRL

---

## 📝 许可证

本示例代码仅供学习和研究使用。

**免责声明**: 本模型不构成投资建议。请自行承担使用本模型进行交易的所有风险。

---

## ✅ 总结

### 核心要点:

1. ✅ 模型已训练并保存为 `ppo_trading_model.zip`
2. ✅ 使用 3 行代码即可加载: `PPO.load()` + `model.predict()`
3. ✅ 动作空间: 0=卖出, 1=持有, 2=买入
4. ✅ 需要 10 个特征作为输入
5. ✅ 可以在新数据上测试和可视化

### 下一步:

- 📖 阅读 `load_ppo_model.py` 了解完整流程
- 🧪 运行测试看模型表现
- 🔧 用真实数据重新训练
- 📊 分析交易策略

祝您交易顺利! 🚀
