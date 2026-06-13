"""
壓縮標題區塊 - 從4行減少到2行
將:
================================================================================
🤖 美股 DECK (Deckers Outdoor) AI 交易信号生成器
================================================================================
生成时间: 2026-02-26 05:46:26
模型準確度: 🟢 AI準確度: 74.0/100
================================================================================

改為:
🤖 DECK (Deckers Outdoor) | 模型準確度: 74.0/100 | 2026-02-26 05:46:26
================================================================================
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import glob
import re

def compact_title_in_file(file_path):
    """壓縮單個檔案的標題區塊"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Pattern: 找到標題區塊並替換
        # 匹配:
        # print("=" * 80)
        # print("🤖 美股 SYMBOL (...) AI 交易信号生成器")
        # print("=" * 80)
        # print(f"生成时间: {datetime.now()...}")
        #
        # accuracy_display = get_model_accuracy_display('SYMBOL')
        # print(f"模型準確度: {accuracy_display}")
        # print("=" * 80)

        pattern = r'(\s+)print\("=" \* 80\)\s*\n\s+print\("🤖 美股 ([A-Z0-9\.]+) \(([^)]+)\) AI 交易信号生成器"\)\s*\n\s+print\("=" \* 80\)\s*\n.*?print\(f?"生成时间:.*?\n\s*\n\s+# 顯示AI模型準確度.*?\n\s+accuracy_display = get_model_accuracy_display\(\'([^\']+)\'\)\s*\n\s+print\(f"模型準確度: \{accuracy_display\}"\)\s*\n\s+print\("=" \* 80\)'

        def replacement(match):
            indent = match.group(1)
            symbol = match.group(2)
            name = match.group(3)
            accuracy_symbol = match.group(4)

            return f'''{indent}# 壓縮標題區塊
{indent}accuracy_display = get_model_accuracy_display('{accuracy_symbol}')
{indent}print(f"🤖 {symbol} ({name}) | 準確度: {{accuracy_display}} | {{datetime.now().strftime('%Y-%m-%d %H:%M')}}")
{indent}print("=" * 80)'''

        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        return False

    except Exception as e:
        print(f"錯誤 {file_path}: {e}")
        return False

# 處理所有信號檔案
signal_files = glob.glob('get_trading_signal_*.py')
total_modified = 0

print("開始壓縮標題區塊...")
print("=" * 80)

for file_path in signal_files:
    if compact_title_in_file(file_path):
        print(f"✅ {file_path}")
        total_modified += 1
    else:
        print(f"⚪ {file_path}: 無需修改或失敗")

print("=" * 80)
print(f"完成! 共修改 {total_modified} 個檔案")
