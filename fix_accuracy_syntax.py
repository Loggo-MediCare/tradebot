"""
修復準確度顯示語法錯誤
"""

import os
import sys
import io
import re
import glob

# 修复 Windows 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def fix_syntax_error(file_path):
    """修復語法錯誤"""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 修復 "})    else:" 的問題
    content = re.sub(r"\)\}\)    else:", ")})\\n    else:", content)

    # 移除重複的準確度顯示
    # 找到所有的重複模式
    accuracy_pattern = r'(\n\s+# 顯示AI模型準確度摘要\n\s+print\(f"   \{get_model_accuracy_display\(.*?\)\}"\)\n)+'

    def replace_duplicates(match):
        """只保留第一次出現"""
        return match.group(1)

    content = re.sub(accuracy_pattern, replace_duplicates, content)

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def main():
    """主程序"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    signal_files = glob.glob(os.path.join(script_dir, 'get_trading_signal_*.py'))

    print("=" * 80)
    print("修復準確度顯示語法錯誤")
    print("=" * 80)
    print(f"檢查 {len(signal_files)} 個信號腳本")
    print()

    fixed_count = 0

    for file_path in signal_files:
        file_name = os.path.basename(file_path)

        try:
            if fix_syntax_error(file_path):
                fixed_count += 1
                print(f"✅ {file_name}")

        except Exception as e:
            print(f"❌ {file_name}: {e}")

    print()
    print("=" * 80)
    print(f"修復完成! 共修復 {fixed_count} 個文件")
    print("=" * 80)


if __name__ == "__main__":
    main()
