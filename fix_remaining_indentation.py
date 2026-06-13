"""
修復剩餘的 FinBERT 縮排錯誤
針對特殊格式的檔案
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import re

def fix_file(file_path):
    """修復單個檔案"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Pattern 1: 修復有兩個 "已註解：避免重複輸出" 的情況
        pattern1 = r"(    if sentiment_result and sentiment_result\['news_count'\] > 0:)\n(    # .*\n)*"

        def replacement1(match):
            return match.group(1) + "\n        pass  # 情緒分析結果已計算，輸出已移至後續統一顯示\n    else:\n        sentiment_result = {'sentiment_score': 0.0, 'news_count': 0, 'sentiment_label': '中性'}\n"

        # 找到並修復
        lines = content.split('\n')
        new_lines = []
        i = 0
        modified = False

        while i < len(lines):
            line = lines[i]

            # 檢測問題模式
            if "if sentiment_result and sentiment_result['news_count'] > 0:" in line:
                # 檢查下一行是否為註解（表示缺少 body）
                if i + 1 < len(lines) and lines[i + 1].strip().startswith('#'):
                    # 這是有問題的區塊
                    new_lines.append(line)
                    new_lines.append("        pass  # 情緒分析結果已計算，輸出已移至後續統一顯示")
                    new_lines.append("    else:")
                    new_lines.append("        sentiment_result = {'sentiment_score': 0.0, 'news_count': 0, 'sentiment_label': '中性'}")

                    # 跳過所有註解行直到遇到非註解
                    i += 1
                    while i < len(lines) and (lines[i].strip().startswith('#') or lines[i].strip() == ''):
                        i += 1

                    modified = True
                    continue
                else:
                    # 正常的 if 語句
                    new_lines.append(line)
                    i += 1
            else:
                new_lines.append(line)
                i += 1

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines))
            return True
        return False

    except Exception as e:
        print(f"錯誤 {file_path}: {e}")
        return False

# 修復特定檔案
files_to_fix = [
    'get_trading_signal_01810.py',
    'get_trading_signal_2330 (2).py',
    'get_trading_signal_3004.py',
    'get_trading_signal_6187 (3).py',
    'get_trading_signal_7805 (2).py'
]

print("修復剩餘的縮排錯誤...")
print("=" * 80)

for file_path in files_to_fix:
    if fix_file(file_path):
        print(f"✅ {file_path}: 已修復")
    else:
        print(f"⚪ {file_path}: 無需修改或失敗")

print("=" * 80)
print("完成!")
