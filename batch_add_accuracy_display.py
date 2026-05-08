"""
批量為所有交易信號腳本添加AI準確度顯示
"""

import os
import sys
import io
import re
import glob

# 修复 Windows 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def update_signal_script_accuracy(file_path):
    """更新單個信號腳本添加準確度顯示"""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    changes_made = []

    # 1. 添加模型準確度模組導入 (在MA50斜率模組導入之後)
    if 'from model_accuracy_tracker import' not in content:
        import_pattern = r'(from ma50_slope_analysis import.*?\n)'
        import_replacement = r'\1# 导入模型准确度追踪器\nfrom model_accuracy_tracker import ModelAccuracyTracker, get_model_accuracy_display\n'
        content = re.sub(import_pattern, import_replacement, content)
        changes_made.append('添加準確度追踪器導入')

    # 2. 在生成時間後添加準確度顯示
    # 需要提取symbol來動態插入
    symbol_match = re.search(r"print\(f\"生成时间:.*?\"\)\n", content)
    if symbol_match and '模型準確度:' not in content:
        # 嘗試從文件名提取股票代號
        filename = os.path.basename(file_path)
        stock_code = filename.replace('get_trading_signal_', '').replace('.py', '')

        # 判斷是台股還是美股
        if stock_code.isdigit():
            symbol = f"{stock_code}.TW"
        else:
            symbol = stock_code.upper()

        accuracy_code = f'''
    # 顯示AI模型準確度
    accuracy_display = get_model_accuracy_display('{symbol}')
    print(f"模型準確度: {{accuracy_display}}")
'''

        # 在生成時間後插入
        time_pattern = r'(print\(f"生成时间:.*?"\)\n)'
        replacement = r'\1' + accuracy_code
        content = re.sub(time_pattern, replacement, content)
        changes_made.append(f'添加準確度顯示 (symbol={symbol})')

    # 3. 在快速摘要中添加準確度
    if 'get_model_accuracy_display' in content:
        # 查找快速摘要的強度輸出
        summary_pattern = r'(print\(f"   强度: \{result\[\'strength\'\]:.2f\}"\)\n)'
        if re.search(summary_pattern, content) and 'AI準確度' not in re.search(r'if result\[\'strength\'\].*?else:', content, re.DOTALL).group() if re.search(r'if result\[\'strength\'\].*?else:', content, re.DOTALL) else '':
            # 從文件名提取symbol
            filename = os.path.basename(file_path)
            stock_code = filename.replace('get_trading_signal_', '').replace('.py', '')

            if stock_code.isdigit():
                symbol = f"{stock_code}.TW"
            else:
                symbol = stock_code.upper()

            summary_accuracy = f"\n\n        # 顯示AI模型準確度摘要\n        print(f\"   {{get_model_accuracy_display('{symbol}')}}\")"

            content = re.sub(summary_pattern, r'\1' + summary_accuracy, content)
            changes_made.append('添加摘要準確度顯示')

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
    print("批量添加AI準確度顯示到交易信號腳本")
    print("=" * 80)
    print(f"找到 {len(signal_files)} 個信號腳本")
    print()

    updated_count = 0
    skipped_count = 0
    failed_files = []

    for file_path in signal_files:
        file_name = os.path.basename(file_path)

        try:
            updated, changes = update_signal_script_accuracy(file_path)

            if updated:
                updated_count += 1
                print(f"✅ {file_name}")
                for change in changes:
                    print(f"   • {change}")
            else:
                skipped_count += 1
                print(f"⏭️  {file_name} (已包含準確度顯示)")

        except Exception as e:
            failed_files.append((file_name, str(e)))
            print(f"❌ {file_name}: {e}")

    # 總結
    print()
    print("=" * 80)
    print("更新完成!")
    print("=" * 80)
    print(f"✅ 成功更新: {updated_count} 個文件")
    print(f"⏭️  跳過: {skipped_count} 個文件 (已包含準確度顯示)")
    print(f"❌ 失敗: {len(failed_files)} 個文件")

    if failed_files:
        print("\n失敗的文件:")
        for file_name, error in failed_files:
            print(f"   • {file_name}: {error}")


if __name__ == "__main__":
    main()
