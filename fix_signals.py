"""
修复信号文件中的格式问题
"""
import os
import re

signal_files = [
    'get_trading_signal_8131.py',
    'get_trading_signal_2330.py',
    'get_trading_signal_2344.py',
    'get_trading_signal_2317.py',
    'get_trading_signal_6770.py',
    'get_trading_signal_goog.py',
    'get_trading_signal_avgo.py',
    'get_trading_signal_2337.py',
    'get_trading_signal_mu.py',
    'get_trading_signal_aapl.py',
    'get_trading_signal_1519.py',
    'get_trading_signal_3017.py',
    'get_trading_signal_3711.py',
    'get_trading_signal_3715.py',
    'get_trading_signal_4938.py',
    'get_trading_signal_6209.py',
    'get_trading_signal_6269.py',
    'get_trading_signal_6443.py',
    'get_trading_signal_6515.py',
    'get_trading_signal_6805.py',
    'get_trading_signal_8210.py',
    'get_trading_signal_nvda.py',
]

for filename in signal_files:
    filepath = os.path.join(r'C:\Users\Silvi\Projects\trading-bot', filename)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复错误的换行
    content = re.sub(r'print\(f"\s*\n\s*{signal_emoji}', r'print(f"\\n{signal_emoji}', content)
    content = re.sub(r'print\(f"\s*\n\s*   📌', r'print(f"\\n   📌', content)
    content = re.sub(r'print\(f"\s*\n\s*   💡', r'print(f"\\n   💡', content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Fixed: {filename}")

print("Done!")
