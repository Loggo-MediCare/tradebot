"""
修復所有準確度顯示相關問題
"""

import sys
import io
import glob
import re

# 修复 Windows 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

files = glob.glob('get_trading_signal_*.py')
fixed_count = 0

print("=" * 80)
print("修復所有準確度顯示問題")
print("=" * 80)
print()

for filepath in files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 1. 修復 ")})    else:" 問題
        content = re.sub(r"\)\}\)\s+else:", ")})\n    else:", content)

        # 2. 移除重複的準確度顯示
        # 匹配重複的準確度顯示塊
        content = re.sub(
            r'(        # 顯示AI模型準確度摘要\n        print\(f"   \{get_model_accuracy_display\([^)]+\)\}"\)\n)(\s*\n\s*# 顯示AI模型準確度摘要\n        print\(f"   \{get_model_accuracy_display\([^)]+\)\}"\)\n)+',
            r'\1',
            content
        )

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            fixed_count += 1
            print(f"✅ 修復: {filepath}")

    except Exception as e:
        print(f"❌ 錯誤 {filepath}: {e}")

print()
print("=" * 80)
print(f"修復完成! 共修復 {fixed_count} 個文件")
print("=" * 80)
