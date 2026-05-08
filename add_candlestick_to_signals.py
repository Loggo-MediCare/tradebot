"""
批次更新所有交易信號腳本，加入蠟燭圖型態分析
"""
import os
import sys
import io
import re
import glob

# 修復Windows編碼問題
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 要插入的import語句
IMPORT_LINE = "from candlestick_patterns import analyze_candlestick_patterns, format_pattern_output, get_pattern_score_adjustment\n"

# 要插入的蠟燭圖分析代碼（插入在市場情緒分析之後）
CANDLESTICK_CODE = """
    # 蠟燭圖型態分析
    print("\\n" + "=" * 80)
    print("📊 蠟燭圖型態分析")
    print("=" * 80)

    try:
        patterns = analyze_candlestick_patterns(df, days=5)
        print(format_pattern_output(patterns))

        # 獲取型態評分調整
        pattern_adjustment = get_pattern_score_adjustment(patterns)
        print(f"\\n   型態評分調整: {pattern_adjustment:+.1f} 分")
    except Exception as e:
        print(f"   ⚠️  型態分析失敗: {e}")
        pattern_adjustment = 0
"""

# 要插入的評分調整代碼（在buy_score計算之後）
SCORE_ADJUSTMENT_CODE = """
        # 加入蠟燭圖型態調整
        buy_score = min(100, max(0, buy_score + pattern_adjustment))
"""


def update_signal_file(filepath):
    """更新單個交易信號文件"""

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 檢查是否已經包含蠟燭圖分析
    if 'candlestick_patterns' in content:
        print(f"   ✓ {os.path.basename(filepath)} - 已包含蠟燭圖分析，跳過")
        return False

    modified = False

    # 1. 添加import語句（在其他import之後，在ImprovedTradingEnv之前）
    if 'from finbert_enhanced_scoring import' in content and 'candlestick_patterns' not in content:
        # 找到finbert import的位置
        finbert_pos = content.find('from finbert_enhanced_scoring import')
        # 找到該行的結尾
        line_end = content.find('\n', finbert_pos)
        # 在下一行插入
        content = content[:line_end+1] + IMPORT_LINE + content[line_end+1:]
        modified = True

    # 2. 添加蠟燭圖分析代碼（在市場情緒分析之後，AI交易信號之前）
    # 尋找 "# 8. 生成交易建议" 或類似的標記
    ai_signal_pattern = r'(    # \d+\. 生成交易[建議建议].*?\n    print\("\n" \+ "=" \* 80\)\n    print\("🎯 AI 交易[信号信號]"\))'

    match = re.search(ai_signal_pattern, content, re.DOTALL)
    if match:
        # 在AI交易信號之前插入蠟燭圖分析
        insert_pos = match.start()
        content = content[:insert_pos] + CANDLESTICK_CODE + '\n' + content[insert_pos:]
        modified = True

    # 3. 添加評分調整代碼（在buy_score計算之後）
    # 尋找buy_score的計算位置
    buy_score_pattern = r'(buy_score, signal_override.*?\n.*?\n.*?\n.*?\))'

    match = re.search(buy_score_pattern, content, re.DOTALL)
    if match:
        insert_pos = match.end()
        # 確保不會重複插入
        if 'pattern_adjustment' not in content[insert_pos:insert_pos+200]:
            content = content[:insert_pos] + SCORE_ADJUSTMENT_CODE + content[insert_pos:]
            modified = True

    if modified:
        # 寫回文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   ✓ {os.path.basename(filepath)} - 成功更新")
        return True
    else:
        print(f"   ⚠️  {os.path.basename(filepath)} - 未找到插入點")
        return False


def main():
    """主函數"""
    print("=" * 80)
    print("批次更新交易信號腳本 - 加入蠟燭圖型態分析")
    print("=" * 80)

    # 找到所有交易信號文件
    signal_files = glob.glob('get_trading_signal_*.py')

    print(f"\n找到 {len(signal_files)} 個交易信號文件\n")

    updated = 0
    skipped = 0
    failed = 0

    for filepath in sorted(signal_files):
        try:
            result = update_signal_file(filepath)
            if result:
                updated += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"   ❌ {os.path.basename(filepath)} - 錯誤: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print("更新完成！")
    print("=" * 80)
    print(f"   ✓ 成功更新: {updated} 個")
    print(f"   ⊘ 已存在: {skipped} 個")
    print(f"   ❌ 失敗: {failed} 個")
    print("=" * 80)


if __name__ == "__main__":
    main()
