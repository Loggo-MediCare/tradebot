"""
移除重複的 FinBERT 輸出區塊
刪除 "📰 市场情绪分析 (FinBERT NLP Engine)" 區塊的 print 輸出
保留情緒分析計算邏輯，只移除顯示輸出
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import glob

def remove_duplicate_finbert_output(file_path):
    """移除重複的 FinBERT 輸出，但保留計算邏輯"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    modifications = 0

    while i < len(lines):
        line = lines[i]

        # 檢測 "📰 市场情绪分析 (FinBERT NLP Engine)" 區塊開始
        if '# 7.5 获取市场情绪分析' in line or '# 7.5 獲取市場情緒分析' in line:
            # 保留註釋行
            new_lines.append(line)
            i += 1

            # 註解掉接下來的 print 區塊，直到遇到 from/import 或實際計算
            while i < len(lines):
                current = lines[i]

                # 如果是 print 開頭的行（含 = 分隔線和情緒標題），註解掉
                if 'print("\\n" + "=" * 80)' in current and i+1 < len(lines) and ('📰 市场情绪分析' in lines[i+1] or '市场情绪分析' in current):
                    # 註解掉分隔線
                    new_lines.append('    # ' + current.lstrip())
                    modifications += 1
                    i += 1
                    # 註解掉標題
                    if i < len(lines) and 'print("📰 市场情绪分析' in lines[i]:
                        new_lines.append('    # ' + lines[i].lstrip())
                        modifications += 1
                        i += 1
                    # 註解掉結束分隔線
                    if i < len(lines) and 'print("=" * 80)' in lines[i]:
                        new_lines.append('    # ' + lines[i].lstrip())
                        modifications += 1
                        i += 1
                    # 註解掉空行
                    if i < len(lines) and lines[i].strip() == '':
                        new_lines.append('    # ' + lines[i].lstrip() if lines[i].strip() else '\n')
                        i += 1
                    continue

                # 如果遇到 from 或 import，保留（這是計算邏輯）
                elif 'from finbert_enhanced_scoring import' in current:
                    new_lines.append(current)
                    i += 1
                    break

                # 其他情況，保留原行
                else:
                    new_lines.append(current)
                    i += 1
                    break
            continue

        # 註解掉獨立的 FinBERT 輸出 print 區塊
        elif 'print("📰 市场情绪分析' in line or 'print("🗞️  市场情绪分析' in line:
            # 檢查是否為第一個區塊（📰）
            if '📰' in line:
                new_lines.append('    # ' + line.lstrip())
                modifications += 1
                i += 1
                continue
            else:
                # 保留第二個區塊（🗞️）
                new_lines.append(line)
                i += 1
                continue

        # 註解掉 print(format_sentiment_output(sentiment_result))
        elif 'print(format_sentiment_output(sentiment_result))' in line:
            new_lines.append('    # ' + line.lstrip() + '    # 已註解：避免重複輸出\n')
            modifications += 1
            i += 1
            continue

        # 註解掉相關的 else 分支（未找到新聞的提示）
        elif i > 0 and 'print(format_sentiment_output(sentiment_result))' in lines[i-1]:
            # 檢查接下來幾行是否為 else 分支
            if 'else:' in line:
                new_lines.append('    # ' + line.lstrip())
                modifications += 1
                i += 1
                # 註解掉 else 裡的 print
                while i < len(lines) and (lines[i].startswith('    ') or lines[i].strip() == ''):
                    if 'print(' in lines[i] or 'sentiment_result' in lines[i]:
                        new_lines.append('    # ' + lines[i].lstrip())
                        modifications += 1
                        i += 1
                        if 'sentiment_result' in lines[i-1] and '}' in lines[i-1]:
                            break
                    else:
                        break
                continue

        # 保留其他所有行
        new_lines.append(line)
        i += 1

    return new_lines, modifications

# 處理所有 get_trading_signal_*.py 檔案
signal_files = glob.glob('get_trading_signal_*.py')
total_files = 0
total_modifications = 0

print("開始移除重複的 FinBERT 輸出區塊...")
print("=" * 80)

for file_path in signal_files:
    new_content, mods = remove_duplicate_finbert_output(file_path)

    if mods > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_content)
        print(f"✅ {file_path}: 移除 {mods} 處輸出")
        total_files += 1
        total_modifications += mods
    else:
        print(f"⚪ {file_path}: 無需修改")

print("=" * 80)
print(f"完成! 共修改 {total_files} 個檔案，移除 {total_modifications} 處重複輸出")
