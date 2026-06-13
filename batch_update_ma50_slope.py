"""
批量更新所有交易信號腳本，加入MA50斜率分析
"""

import os
import sys
import io
import re
import glob

# 修复 Windows 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def update_signal_script(file_path):
    """更新單個信號腳本"""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    changes_made = []

    # 1. 添加MA50斜率模組導入 (在candlestick_patterns導入之後)
    if 'from ma50_slope_analysis import' not in content:
        import_pattern = r'(from candlestick_patterns import.*?\n)'
        import_replacement = r'\1# 导入MA50斜率分析模块\nfrom ma50_slope_analysis import calculate_ma50_slope, format_ma50_slope_output, get_ma50_slope_score_adjustment\n'
        content = re.sub(import_pattern, import_replacement, content)
        changes_made.append('添加MA50斜率模組導入')

    # 2. 在技術指標分析後添加MA50趨勢分析
    if 'MA50趨勢分析' not in content:
        # 找到量比輸出的位置
        volume_pattern = r'(print\(f"量比:\s+\{volume_ratio:.2f\}x"\))'
        ma50_analysis_code = r'''\1

    # 7.1 計算MA50斜率
    print("\\n" + "=" * 80)
    print("📈 MA50趨勢分析")
    print("=" * 80)
    ma50_slope_info = calculate_ma50_slope(df['close'], window=50, slope_period=5)
    print(f"當前MA50:        NT${ma50_slope_info['ma50_current']:.2f}")
    print(f"MA50斜率:        {ma50_slope_info['slope']:+.6f}")
    print(f"斜率百分比:      {ma50_slope_info['slope_pct']:+.4f}%")
    print(f"趨勢判斷:        {ma50_slope_info['color']} {ma50_slope_info['trend']}")
    print(f"交易信號:        {ma50_slope_info['signal']}")
    print(f"\\n💡 說明: {ma50_slope_info['description']}")'''

        content = re.sub(volume_pattern, ma50_analysis_code, content)
        changes_made.append('添加MA50趨勢分析輸出')

    # 3. 在買入評分系統後添加MA50斜率調整
    if 'ma50_slope_adjustment = get_ma50_slope_score_adjustment' not in content:
        # 找到買入評分計算的位置
        buy_score_pattern = r'(buy_score, signal_override, buy_reasons, buy_warnings, buy_metadata, sentiment_result = calculate_enhanced_buy_score_with_sentiment\(.*?\))'
        buy_adjustment_code = r'''\1

        # 加入MA50斜率評分調整
        ma50_slope_adjustment = get_ma50_slope_score_adjustment(ma50_slope_info)
        buy_score += ma50_slope_adjustment
        buy_score = max(0, min(100, buy_score))  # 限制在0-100之間

        if ma50_slope_adjustment > 0:
            buy_reasons.append(f"MA50趨勢向上 (+{ma50_slope_adjustment}分)")
        elif ma50_slope_adjustment < 0:
            buy_warnings.append(f"MA50趨勢向下 ({ma50_slope_adjustment}分)")'''

        content = re.sub(buy_score_pattern, buy_adjustment_code, content, flags=re.DOTALL)
        changes_made.append('添加買入評分MA50調整')

    # 4. 在賣出評分系統中添加MA50斜率調整
    if '# 加入MA50斜率評分調整 (負斜率增加賣出分數)' not in content:
        # 找到賣出評分初始化的位置
        sell_score_pattern = r'(# 卖出信号评分系统（0-100分）\s+sell_score = 0\s+reasons = \[\])'
        sell_adjustment_code = r'''\1

        # 加入MA50斜率評分調整 (負斜率增加賣出分數)
        ma50_slope_adjustment = get_ma50_slope_score_adjustment(ma50_slope_info)
        if ma50_slope_adjustment < 0:
            sell_score += abs(ma50_slope_adjustment)
            reasons.append(f"MA50趨勢向下 ({ma50_slope_info['slope_pct']:.2f}%)")
        elif ma50_slope_adjustment > 0:
            # MA50向上趨勢，降低賣出評分
            sell_score -= abs(ma50_slope_adjustment) * 0.5  # 使用0.5係數減少影響
            if sell_score < 0:
                sell_score = 0'''

        content = re.sub(sell_score_pattern, sell_adjustment_code, content)
        changes_made.append('添加賣出評分MA50調整')

    # 5. 在技術指標顯示中添加SMA 50的輸出（如果沒有的話）
    if 'SMA 50:' not in content and 'SMA 30:' in content:
        sma30_pattern = r'(print\(f"SMA 30:\s+NT\$\{sma_30:.2f\}.*?"\))'
        sma50_display = r'\1\n    print(f"SMA 50:          NT${sma_30:.2f}")'
        content = re.sub(sma30_pattern, sma50_display, content)
        changes_made.append('添加SMA 50顯示')

    # 只有在有變更時才寫入
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, changes_made
    else:
        return False, []


def main():
    """主程序"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    signal_files = glob.glob(os.path.join(script_dir, 'get_trading_signal_*.py'))

    print("=" * 80)
    print("批量更新交易信號腳本 - 添加MA50斜率分析")
    print("=" * 80)
    print(f"找到 {len(signal_files)} 個信號腳本")
    print()

    updated_count = 0
    skipped_count = 0
    failed_files = []

    for file_path in signal_files:
        file_name = os.path.basename(file_path)

        try:
            updated, changes = update_signal_script(file_path)

            if updated:
                updated_count += 1
                print(f"✅ {file_name}")
                for change in changes:
                    print(f"   • {change}")
            else:
                skipped_count += 1
                print(f"⏭️  {file_name} (已包含MA50分析)")

        except Exception as e:
            failed_files.append((file_name, str(e)))
            print(f"❌ {file_name}: {e}")

    # 總結
    print()
    print("=" * 80)
    print("更新完成!")
    print("=" * 80)
    print(f"✅ 成功更新: {updated_count} 個文件")
    print(f"⏭️  跳過: {skipped_count} 個文件 (已包含MA50分析)")
    print(f"❌ 失敗: {len(failed_files)} 個文件")

    if failed_files:
        print("\n失敗的文件:")
        for file_name, error in failed_files:
            print(f"   • {file_name}: {error}")


if __name__ == "__main__":
    main()
