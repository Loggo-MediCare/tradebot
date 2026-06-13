"""
移除所有1303關聯性檢查（包括AXON、SMCI等）
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import re

FILES_TO_CLEAN = [
    'get_trading_signal_axon.py',
    'get_trading_signal_smci.py',
]

def remove_1303_correlation(file_path):
    """移除1303關聯性檢查"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Pattern: 從 "# 1. 检查与1303的相关性" 到下一個 "# 2."
        pattern = r'        # 1\. 检查与1303的相关性.*?(?=        # 2\.)'

        new_content = re.sub(pattern, '        # 1303相关性检查已移除\n\n', content, flags=re.DOTALL)

        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        return False

    except Exception as e:
        print(f"錯誤 {file_path}: {e}")
        return False

# 處理所有需要移除的檔案
total_modified = 0

print("開始移除所有1303關聯性檢查...")
print("=" * 80)

for file_path in FILES_TO_CLEAN:
    if remove_1303_correlation(file_path):
        print(f"✅ {file_path}")
        total_modified += 1
    else:
        print(f"⚪ {file_path}: 無需修改或失敗")

print("=" * 80)
print(f"完成! 共修改 {total_modified} 個檔案")
