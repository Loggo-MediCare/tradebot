"""
Add explosion detection usage to get_trading*.py files that have the functions but not the usage
"""
import os
import re
import sys
import io
from pathlib import Path

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def add_explosion_detection_usage(file_path):
    """Add explosion detection usage if function exists but usage doesn't"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if function exists
        if 'def explosive_trend_filter' not in content:
            return False, "Function not found"

        # Check if usage already exists
        if 'explosion = explosive_trend_filter(df)' in content:
            return True, "Already has usage"

        original_content = content

        # Find the position to insert explosion detection (before signal generation)
        # Look for the "AI 交易信号" or "交易建议" section
        patterns = [
            r'(\n    # \d+\. 生成交易建议\s+print\("\\n" \+ "=" \* 80\)\s+print\("🎯 AI 交易信号"\))',
            r'(\n    # \d+\. AI 交易信号\s+print\("\\n" \+ "=" \* 80\)\s+print\("🎯 AI 交易信号"\))',
            r'(\n    print\("\\n" \+ "=" \* 80\)\s+print\("🎯 AI 交易信号"\)\s+print\("=" \* 80\))',
        ]

        insertion_point = None
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                insertion_point = match.start()
                break

        if not insertion_point:
            return False, "Could not find insertion point"

        # Insert explosion detection code
        explosion_code = '''
    # 爆发行情检测（主升段分析）
    print("\\n" + "=" * 80)
    print("🚀 爆发行情检测 (主升段分析)")
    print("=" * 80)

    explosion = explosive_trend_filter(df)
    print(f"资金流入状态: {'✅ 强势' if explosion['money_inflow'] else '❌ 弱势'}")
    print(f"趋势加速状态: {'✅ 加速中' if explosion['trend_accelerating'] else '❌ 减速中'}")
    print(f"周期阶段: {explosion['cycle_phase']}")
    print(f"量比: {explosion['volume_ratio']:.2f}x")

    if explosion["explosive"]:
        print("\\n🚀 主升段爆发行情侦测!")
        print("📌 爆发行情特征:")
        print("   • 资金强势流入 (OBV > 20日均线)")
        print("   • 趋势加速 (10日均线斜率 > 30日均线斜率)")
        print("   • 处于周期初升段 (EARLY_UPCYCLE)")
        print("   • 量能放大 (量比 > 1.3x)")
'''

        content = content[:insertion_point] + explosion_code + content[insertion_point:]

        # Now add explosion bonus to buy signals
        buy_pattern = r'(        buy_score \+= ma50_slope_adjustment\s+)'

        if re.search(buy_pattern, content):
            explosion_bonus = '''

        # 加入爆发行情评分调整
        if explosion["explosive"]:
            buy_score += 25  # 爆发行情额外加分
            buy_reasons.append(f"🚀 爆发行情确认: 主升段初期")
            buy_reasons.append(f"资金强势流入 (OBV > MA20)")
            buy_reasons.append(f"趋势加速 (10日斜率 > 30日斜率)")
'''
            content = re.sub(buy_pattern, r'\1' + explosion_bonus, content, count=1)

        # Add explosion override for sell signals
        sell_pattern = r'(\n    elif action_value < -0\.1:\s+signal = "卖出 \(SELL\)")'

        if re.search(sell_pattern, content):
            explosion_override = '''
    elif action_value < -0.1:
        # 先检查是否为爆发行情，如果是则覆盖卖出信号
        if explosion["explosive"]:
            signal = "强势持有 (HOLD - TREND EXPLOSION)"
            signal_emoji = "🚀"
            strength = abs(action_value)
            suggested_price_low = current_price
            suggested_price_high = current_price

            print("\\n🚀 主升段爆发行情侦测!")
            print(f"资金流入: {explosion['money_inflow']}")
            print(f"趋势加速: {explosion['trend_accelerating']}")
            print(f"周期位置: {explosion['cycle_phase']}")
            print(f"量比: {explosion['volume_ratio']:.2f}x")

            print("\\n📌 操作策略:")
            print("   • 不卖出 (主升段爆发行情)")
            print("   • 回调不破均线继续抱")
            print("   • 使用追踪止损代替固定止损")
            print("   • 关注 OBV 资金流向指标")
            print("   • 设置移动止盈: 跌破 10 日均线减半仓")

            # 跳过卖出评分逻辑
            skip_sell_scoring = True
        else:
            skip_sell_scoring = False

        if not skip_sell_scoring:
            signal = "卖出 (SELL)"'''

            content = re.sub(sell_pattern, explosion_override, content, count=1)

        # Add explosion_detected to return statement
        return_pattern = r"(return \{[^}]*'strength': abs\(action_value\)[^}]*\})"

        if re.search(return_pattern, content, re.DOTALL):
            # Find the return statement and add explosion_detected
            if "'explosion_detected':" not in content:
                content = re.sub(
                    r"(\s+)(return \{\s+[^}]*\})",
                    lambda m: m.group(1) + m.group(2).replace(
                        "}",
                        ",\n        'explosion_detected': explosion['explosive'] if 'explosion' in locals() else False\n    }"
                    ),
                    content,
                    count=1
                )

        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Successfully added usage"
        else:
            return False, "No changes made"

    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main function"""
    print("=" * 80)
    print("Adding explosion detection usage to all get_trading*.py files")
    print("=" * 80)

    # Find all get_trading*.py files
    current_dir = Path('.')
    files = list(current_dir.glob('get_trading_signal_*.py'))

    if not files:
        print("[ERROR] No get_trading_signal_*.py files found")
        return

    print(f"Found {len(files)} files to process\n")

    success_count = 0
    skip_count = 0
    error_count = 0

    for file_path in sorted(files):
        print(f"Processing {file_path.name}...", end=" ")
        success, message = add_explosion_detection_usage(file_path)

        if success:
            if "Already has" in message:
                print(f"[SKIP] {message}")
                skip_count += 1
            else:
                print(f"[OK] {message}")
                success_count += 1
        else:
            print(f"[ERROR] {message}")
            error_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total files:        {len(files)}")
    print(f"Successfully added: {success_count}")
    print(f"Already had usage:  {skip_count}")
    print(f"Errors:             {error_count}")
    print("=" * 80)

    if success_count > 0:
        print(f"\n[OK] Successfully added explosion detection usage to {success_count} files!")
    else:
        print("\n[INFO] All files already have explosion detection usage or encountered errors")

if __name__ == "__main__":
    main()
