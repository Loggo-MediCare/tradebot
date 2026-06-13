"""
修復 FinBERT 移除後的縮排錯誤
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import glob

def fix_indentation_error(file_path):
    """修復 if sentiment_result 後的縮排問題"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 找到有問題的區塊並修復
    # Pattern: if sentiment_result and sentiment_result['news_count'] > 0:
    #          # print(format_sentiment_output(sentiment_result))
    #          # 已註解：避免重複輸出

    # 替換為正確的版本
    old_pattern1 = """    if sentiment_result and sentiment_result['news_count'] > 0:
    # print(format_sentiment_output(sentiment_result))
    # 已註解：避免重複輸出
    # else:
    # print("⚠️  未找到相关新闻，情绪分析不可用")
    # sentiment_result = {'sentiment_score': 0.0, 'news_count': 0, 'sentiment_label': '中性'}"""

    new_pattern1 = """    if sentiment_result and sentiment_result['news_count'] > 0:
        pass  # 情緒分析結果已計算，輸出已移至後續統一顯示
    else:
        sentiment_result = {'sentiment_score': 0.0, 'news_count': 0, 'sentiment_label': '中性'}"""

    # 處理中性的情況（有些檔案可能不同）
    old_pattern2 = """    if sentiment_result and sentiment_result['news_count'] > 0:
    # print(format_sentiment_output(sentiment_result))
    # 已註解：避免重複輸出
    # else:
    # print("⚠️  未找到相關新聞，情緒分析不可用")
    # sentiment_result = {'sentiment_score': 0.0, 'news_count': 0, 'sentiment_label': '中性'}"""

    modified = False
    if old_pattern1 in content:
        content = content.replace(old_pattern1, new_pattern1)
        modified = True
    elif old_pattern2 in content:
        content = content.replace(old_pattern2, new_pattern1)
        modified = True

    return content, modified

# 處理所有 get_trading_signal_*.py 檔案
signal_files = glob.glob('get_trading_signal_*.py')
total_fixed = 0

print("開始修復 FinBERT 縮排錯誤...")
print("=" * 80)

for file_path in signal_files:
    try:
        new_content, modified = fix_indentation_error(file_path)

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"✅ {file_path}: 已修復")
            total_fixed += 1
        else:
            # 檢查是否已經是正確格式
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if 'pass  # 情緒分析結果已計算' in content:
                print(f"⚪ {file_path}: 已是正確格式")
            else:
                print(f"⚪ {file_path}: 無需修改")
    except Exception as e:
        print(f"❌ {file_path}: 錯誤 - {e}")

print("=" * 80)
print(f"完成! 共修復 {total_fixed} 個檔案")
