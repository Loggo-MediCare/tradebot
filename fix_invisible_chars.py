"""
修复信号文件中的不可见字符
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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

print("=" * 70)
print("修复信号文件中的不可见字符")
print("=" * 70)

for filename in signal_files:
    filepath = os.path.join(r'C:\Users\Silvi\Projects\trading-bot', filename)

    if not os.path.exists(filepath):
        continue

    try:
        # 以二进制模式读取
        with open(filepath, 'rb') as f:
            content = f.read()

        # 检查是否有不可见字符
        if b'\x01' in content:
            # 移除 \x01 字符
            content = content.replace(b'\x01', b'')

            # 写回文件
            with open(filepath, 'wb') as f:
                f.write(content)

            print(f"✅ 修复: {filename}")
        else:
            print(f"⏭️  OK: {filename}")

    except Exception as e:
        print(f"❌ 错误 {filename}: {e}")

print("\n" + "=" * 70)
print("完成!")
print("=" * 70)
