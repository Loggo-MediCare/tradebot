"""
壓縮所有信號檔案的標題區塊
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import glob
import re

def compact_title(file_path):
    """壓縮標題區塊"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        i = 0
        modified = False

        while i < len(lines):
            line = lines[i]

            # 檢測標題區塊開始
            if 'def get_trading_signal():' in line or 'def get_trading_signal(' in line:
                new_lines.append(line)
                i += 1

                # 跳過 docstring
                if i < len(lines) and '"""' in lines[i]:
                    new_lines.append(lines[i])
                    i += 1

                # 檢查是否為舊格式標題區塊
                if i < len(lines) and 'print("=" * 80)' in lines[i]:
                    # 舊格式，需要替換
                    # 跳過所有舊標題行，直到找到實際內容
                    old_block_end = i
                    symbol = None
                    name = None

                    # 掃描並提取symbol和name
                    for j in range(i, min(i+15, len(lines))):
                        if '🤖 美股' in lines[j] or '🤖 台股' in lines[j]:
                            # 提取 symbol 和 name
                            match = re.search(r'[🤖]\s*(美股|台股)\s+([A-Z0-9\.]+)\s*\(([^)]+)\)', lines[j])
                            if match:
                                symbol = match.group(2)
                                name = match.group(3)

                        if 'print("=" * 80)' in lines[j] and j > i + 3:
                            # 找到結束的分隔線
                            old_block_end = j + 1
                            break

                    if symbol:
                        # 插入新的壓縮標題
                        indent = '    '
                        new_lines.append(f"{indent}# 壓縮標題區塊 - 從6行減少到2行\n")
                        new_lines.append(f"{indent}accuracy_display = get_model_accuracy_display('{symbol}')\n")
                        new_lines.append(f"{indent}print(f\"🤖 {symbol} ({name}) | 準確度: {{accuracy_display}} | {{datetime.now().strftime('%Y-%m-%d %H:%M')}}\")\n")
                        new_lines.append(f"{indent}print(\"=\" * 80)\n")

                        i = old_block_end
                        modified = True
                        continue

            new_lines.append(line)
            i += 1

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            return True

        return False

    except Exception as e:
        print(f"錯誤 {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

# 處理所有信號檔案
signal_files = sorted(glob.glob('get_trading_signal_*.py'))
total_modified = 0

print("開始壓縮標題區塊...")
print("=" * 80)

for file_path in signal_files:
    if file_path == 'get_trading_signal_deck.py':
        print(f"⚪ {file_path}: 已手動修改")
        continue

    if compact_title(file_path):
        print(f"✅ {file_path}")
        total_modified += 1
    else:
        print(f"⚪ {file_path}: 無需修改或失敗")

print("=" * 80)
print(f"完成! 共修改 {total_modified} 個檔案")
