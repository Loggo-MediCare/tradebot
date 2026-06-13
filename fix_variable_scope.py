"""
Fix variable scope issue in explosion detection
Variables need to be defined before conditional blocks
"""
import re
from pathlib import Path

def fix_variable_scope(file_path):
    """Fix variable scope in sell scoring section"""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern: variables defined inside if not skip_sell_scoring block
    # Need to move bb_position, is_macd_bearish, is_trending_down BEFORE the if statement

    old_pattern = r'''(\s+)if not skip_sell_scoring:
(\s+)signal = "卖出 \(SELL\)"
(\s+)signal_emoji = "🔴"
(\s+)strength = abs\(action_value\)
(\s+)suggested_price_low = current_price \* 1\.000
(\s+)suggested_price_high = current_price \* 1\.005

(\s+)# 🔥 改进的卖出判断逻辑
(\s+)# 计算更多技术指标
(\s+)bb_position = \(current_price - bb_lower\) / \(bb_upper - bb_lower\) \* 100 if \(bb_upper - bb_lower\) > 0 else 50
(\s+)is_macd_bearish = macd < macd_signal
(\s+)is_trending_down = sma_10 < sma_30

(\s+)# 卖出信号评分系统'''

    new_pattern = r'''\1# 🔥 计算技术指标（需要在条件块外定义，以便后续使用）
\1bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
\1is_macd_bearish = macd < macd_signal
\1is_trending_down = sma_10 < sma_30

\1if not skip_sell_scoring:
\2signal = "卖出 (SELL)"
\3signal_emoji = "🔴"
\4strength = abs(action_value)
\5suggested_price_low = current_price * 1.000
\6suggested_price_high = current_price * 1.005

\12# 卖出信号评分系统'''

    # Check if pattern exists
    if re.search(old_pattern, content):
        content = re.sub(old_pattern, new_pattern, content)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True

    return False

def main():
    """Fix variable scope in all signal files"""
    print("=" * 80)
    print("Fixing Variable Scope Issues")
    print("=" * 80)

    signal_files = sorted(Path('.').glob('get_trading_signal_*.py'))

    fixed_count = 0
    skipped_count = 0

    for file_path in signal_files:
        try:
            if fix_variable_scope(file_path):
                print(f"[OK] Fixed {file_path.name}")
                fixed_count += 1
            else:
                print(f"[SKIP] {file_path.name} - pattern not found or already fixed")
                skipped_count += 1
        except Exception as e:
            print(f"[ERROR] {file_path.name}: {e}")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Fixed: {fixed_count} files")
    print(f"Skipped: {skipped_count} files")
    print(f"Total: {len(signal_files)} files")
    print("=" * 80)
    print("\n✅ Variable scope issue fixed!")
    print("Variables bb_position, is_macd_bearish, is_trending_down")
    print("are now defined BEFORE the if not skip_sell_scoring block")
    print("=" * 80)

if __name__ == "__main__":
    main()
