"""
从已有的训练输出中提取特征重要性数据并保存为 JSON
用于测试动态权重系统
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
from datetime import datetime

# NVDA 特征重要性 (从之前的训练输出)
nvda_importance = {
    'MA_200': 0.067242,
    'MA_20': 0.065720,
    'macd_signal': 0.065078,
    'MA50_slope': 0.064566,
    'ATR': 0.064470,
    'OBV_MA': 0.064123,
    'MA_50': 0.063923,
    'volatility': 0.063323,
    'macd_hist': 0.058402,
    'OBV': 0.058295,
    'price_change_20d': 0.058254,
    'macd': 0.058129,
    'price_change_5d': 0.053596,
    'D': 0.051464,
    'rsi': 0.050381,
    'K': 0.046643,
    'bb_position': 0.046390,
}

# 2330.TW 特征重要性 (从用户提供的数据)
tw2330_importance = {
    'MA_200': 0.134121,
    'MA_50': 0.087383,
    'OBV_MA': 0.085104,
    'MA50_slope': 0.077356,
    'MA_20': 0.075371,
    'OBV': 0.073152,
    'volatility': 0.065177,
    'ATR': 0.059577,
    'macd_signal': 0.055222,  # 注意：用户数据中是 MACD_Signal
    'price_change_20d': 0.051338,
    'macd': 0.046571,
    'macd_hist': 0.039468,
    'D': 0.033587,
    'bb_position': 0.030914,
    'rsi': 0.029764,  # RSI 最不重要！
}

stocks = [
    ('NVDA', nvda_importance, 0.5234),  # (ticker, importance_dict, accuracy)
    ('2330.TW', tw2330_importance, 0.5156),
]

print("=" * 70)
print("提取特征重要性数据并保存为 JSON")
print("=" * 70)

for ticker, importance_dict, accuracy in stocks:
    json_data = {
        'ticker': ticker,
        'analysis_date': datetime.now().strftime('%Y-%m-%d'),
        'model_accuracy': accuracy,
        'feature_importance': importance_dict
    }

    filename = f"{ticker}_feature_importance.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 已保存: {filename}")
    print(f"   准确率: {accuracy:.4f}")
    print(f"   特征数: {len(importance_dict)}")
    print()

print("=" * 70)
print("完成！现在可以测试动态权重系统了")
print("=" * 70)
