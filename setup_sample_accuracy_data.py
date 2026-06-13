#!/usr/bin/env python3
"""
設置示例準確度數據
"""
import sys
import io

# 修复 Windows 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from model_accuracy_tracker import ModelAccuracyTracker

# 為主要台股創建示例數據
stocks_tw = [
    ('2344.TW', 82.3, 76.8, 71.5, 68.2, 1.65),
    ('2330.TW', 85.1, 80.3, 75.2, 72.1, 1.92),
    ('2317.TW', 78.9, 73.5, 69.3, 66.8, 1.58),
    ('2454.TW', 81.2, 75.9, 70.8, 67.5, 1.72),
    ('3443.TW', 83.5, 77.2, 72.1, 69.3, 1.78),
]

# 為主要美股創建示例數據
stocks_us = [
    ('HTGC', 79.5, 74.2, 69.8, 65.5, 1.52),
    ('NVDA', 84.2, 78.9, 73.5, 70.2, 1.85),
    ('TSLA', 76.8, 71.5, 67.2, 63.8, 1.45),
    ('AAPL', 82.1, 76.5, 71.3, 68.0, 1.68),
]

print("=" * 80)
print("設置AI模型準確度示例數據")
print("=" * 80)
print()

for symbol, train_acc, val_acc, backtest_acc, win_rate, sharpe in stocks_tw + stocks_us:
    tracker = ModelAccuracyTracker(symbol)
    tracker.update_training_stats(
        training_acc=train_acc,
        validation_acc=val_acc,
        backtest_acc=backtest_acc,
        win_rate=win_rate,
        sharpe_ratio=sharpe
    )
    print(f"✅ 已為 {symbol} 創建準確度數據 (回測: {backtest_acc}%, 勝率: {win_rate}%)")

print()
print("=" * 80)
print("設置完成!")
print("=" * 80)
