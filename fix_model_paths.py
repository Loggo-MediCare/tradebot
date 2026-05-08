"""
修复所有 get_trading_signal 文件中的模型路径 (.zip 扩展名问题)
"""
import os
import sys
import io
import re

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 获取所有 get_trading_signal 文件
files = [f for f in os.listdir('.') if f.startswith('get_trading_signal') and f.endswith('.py')]

fixed_count = 0

for filename in files:
    filepath = os.path.join('.', filename)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 查找并替换 model_path 中的 .zip 扩展名
    # 匹配: model_path = r"...\ppo_xxx_improved.zip"
    pattern = r'(model_path = r"[^"]+)\.zip(")'

    if re.search(pattern, content):
        new_content = re.sub(pattern, r'\1\2', content)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✅ 已修复: {filename}")
        fixed_count += 1
    else:
        print(f"⏭️  跳过: {filename} (未找到需要修复的路径)")

print(f"\n🎉 完成! 共修复 {fixed_count} 个文件")
