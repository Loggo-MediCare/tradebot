"""
批次更新交易信號檔案改用 XGBoost 模型
將 PPO 模型替換為 XGBoost 模型
"""
import re
import os

# 需要更新的股票列表 (排除 3138 因為準確度反而下降)
stocks_to_update = [
    {'code': '3004', 'exchange': 'TW', 'name': '豐達科'},
    {'code': '3135', 'exchange': 'TW', 'name': ''},
    {'code': '2357', 'exchange': 'TW', 'name': '華碩'},
    {'code': '3363', 'exchange': 'TWO', 'name': '上詮'},
    {'code': '8069', 'exchange': 'TWO', 'name': '元太'},
]

# XGBoost 版本的 imports
new_imports = '''"""
台股 {code} {name} AI 交易信号生成器
======================================
使用训练好的 XGBoost 模型生成今日交易策略
输出: 买入/卖出/持有 信号 + 建议价格
"""

import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
'''

# XGBoost 版本的技術指標函數
new_technical_indicators = '''# ==========================================
# 技术指标计算 (XGBoost 所需特征)
# ==========================================
def add_technical_indicators(df):
    """添加技术指标 - 包含 XGBoost 模型所需的所有特征"""
    # 均线
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']  # XGBoost 需要

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100).fillna(50)  # XGBoost 需要

    # KD 指标 (XGBoost 需要)
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['K'] = ((df['close'] - low_14) / (high_14 - low_14) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()

    # OBV (能量潮指标)
    df = calculate_obv(df)

    # 波動性 (XGBoost 需要)
    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()

    # ATR (XGBoost 需要)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()

    # 價格變化率 (XGBoost 需要)
    df['price_change_5d'] = df['close'].pct_change(5) * 100
    df['price_change_10d'] = df['close'].pct_change(10) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100

    # MA50 斜率 (XGBoost 需要)
    df['ma50_slope'] = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100

    df = df.bfill().ffill()
    return df
'''

# XGBoost 模型載入代碼
def get_model_loading_code(code, exchange):
    model_suffix = 'two' if exchange == 'TWO' else 'tw'
    return f'''    # 1. 加载 XGBoost 模型
    model_filename = "xgb_{code}_{model_suffix}_model.pkl"
    print(f"\\n📦 加载 XGBoost 模型: {{model_filename}}")

    try:
        model = joblib.load(model_filename)
        print("✅ XGBoost 模型加载成功!")
    except Exception as e:
        print(f"❌ 模型加载失败: {{e}}")
        return None'''

# XGBoost 預測代碼
xgboost_prediction_code = '''    # 4. 准备 XGBoost 模型特征
    feature_columns = [
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_position', 'K', 'D', 'obv', 'obv_ma20',
        'sma_10', 'sma_30', 'sma_50', 'sma_200',
        'volatility', 'atr',
        'price_change_5d', 'price_change_10d', 'price_change_20d',
        'ma50_slope'
    ]

    # 5. 使用 XGBoost 模型预测
    print("\\n🧠 XGBoost 模型分析中...")
    latest_features = df[feature_columns].iloc[[-1]]  # 获取最新数据的特征

    # XGBoost 预测 (0=不买, 1=买入)
    prediction = model.predict(latest_features)[0]
    prediction_proba = model.predict_proba(latest_features)[0]

    # 将 XGBoost 预测转换为 action_value (-1.0 到 1.0)
    # prediction=1 (买入) -> action_value > 0
    # prediction=0 (不买) -> action_value < 0
    if prediction == 1:
        # 买入信号，根据概率决定强度
        action_value = float(prediction_proba[1] * 2 - 1)  # 转换为 0 到 1 之间
        action_value = max(0.1, action_value)  # 确保至少 0.1
    else:
        # 不买/卖出信号
        action_value = -float(prediction_proba[0] * 2 - 1)  # 转换为 -1 到 0 之间
        action_value = min(-0.1, action_value)  # 确保至少 -0.1

    print(f"   XGBoost 预测: {{'买入' if prediction == 1 else '不买/观望'}}")
    print(f"   买入概率: {{prediction_proba[1]*100:.2f}}%")
    print(f"   Action Value: {{action_value:+.4f}}")

    ai_muted = should_mute_ai_signal('{symbol}', threshold=52)
    if ai_muted:
        print("⚠️  AI模型準確度低於52%，已靜音AI交易動作（action=0）")
        action_value = 0.0'''

def update_signal_file(stock):
    code = stock['code']
    exchange = stock['exchange']
    name = stock['name']
    symbol = f"{code}.{exchange}"

    filename = f"get_trading_signal_{code}.py"

    if not os.path.exists(filename):
        print(f"⚠️  檔案不存在: {filename}")
        return False

    print(f"\n{'='*80}")
    print(f"更新: {code} {name} ({symbol})")
    print(f"{'='*80}")

    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 替換 imports
    # 找到第一個 import 前的 docstring
    docstring_match = re.search(r'^""".*?"""', content, re.DOTALL | re.MULTILINE)
    if docstring_match:
        # 找到所有 imports 結束的位置 (第一個不是 import/from/# 的行)
        imports_end = content.find('\n\n# ===', docstring_match.end())
        if imports_end > 0:
            # 替換整個 imports 區塊
            new_content = content[:docstring_match.start()] + new_imports.format(code=code, name=name) + content[imports_end:]
            content = new_content

    # 2. 移除 ImprovedTradingEnv class
    env_class_pattern = r'# ={40,}\n# 交易环境.*?\n# ={40,}\nclass ImprovedTradingEnv.*?return obs, float\(reward\), done, False, \{\}\n\n'
    content = re.sub(env_class_pattern, '', content, flags=re.DOTALL)

    # 3. 替換技術指標函數
    tech_indicators_pattern = r'# ={40,}\n# 技术指标计算.*?\n# ={40,}\ndef add_technical_indicators\(df\):.*?return df\n\n'
    content = re.sub(tech_indicators_pattern, new_technical_indicators + '\n\n', content, flags=re.DOTALL)

    # 4. 替換模型載入
    model_loading_pattern = r'    # 1\. 加载模型.*?return None\n'
    model_loading_code = get_model_loading_code(code, exchange)
    content = re.sub(model_loading_pattern, model_loading_code + '\n', content, flags=re.DOTALL)

    # 5. 替換預測邏輯
    prediction_pattern = r'    # 4\. 创建环境并获取观察值.*?action_value = 0\.0\n'
    prediction_code = xgboost_prediction_code.replace('{symbol}', symbol)
    content = re.sub(prediction_pattern, prediction_code + '\n', content, flags=re.DOTALL)

    # 寫回檔案
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ 已更新: {filename}")
    return True

# 主程序
if __name__ == '__main__':
    print("="*80)
    print("批次更新交易信號檔案改用 XGBoost 模型")
    print("="*80)

    success_count = 0
    for stock in stocks_to_update:
        if update_signal_file(stock):
            success_count += 1

    print(f"\n{'='*80}")
    print(f"更新完成!")
    print(f"{'='*80}")
    print(f"成功: {success_count}/{len(stocks_to_update)}")
    print(f"{'='*80}")
