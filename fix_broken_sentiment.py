"""
修复被错误插入的情绪分析代码
"""
import os
import sys
import io
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def fix_file(filepath):
    """修复文件中的错误语法"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 查找并修复错误的 print 语句
    # 错误: print("\n" (被分成两行)
    # 正确: print("\n" + "=" * 80)

    fixed = re.sub(
        r'print\("[\r\n]+"\s*\+\s*"="\s*\*\s*80\)',
        r'print("\\n" + "=" * 80)',
        content
    )

    if fixed != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed)
        return True
    return False

# 获取所有信号文件
files = [f for f in os.listdir('.') if f.startswith('get_trading_signal_') and f.endswith('.py')]

print(f"检查 {len(files)} 个文件...")
fixed_count = 0

for filename in sorted(files):
    filepath = os.path.join('.', filename)
    if fix_file(filepath):
        print(f"✅ 修复: {filename}")
        fixed_count += 1

print(f"\n完成! 修复了 {fixed_count} 个文件")
