"""
Add explosion detection to final summary in all get_trading*.py files
"""
import os
import re
import sys
import io
from pathlib import Path

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def add_explosion_to_summary(file_path):
    """Add explosion detection to final summary"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes_made = []

        # Step 1: Add explosion_detected to return dictionary if not present
        if "'explosion_detected':" not in content:
            # Find the return dictionary
            pattern = r"(    return \{[^}]*'suggested_price_high': [^,\n]+)"
            match = re.search(pattern, content, re.DOTALL)

            if match:
                # Add explosion fields before the closing brace
                replacement = r"\1,\n        'explosion_detected': explosion[\"explosive\"] if 'explosion' in locals() else False,\n        'explosion_data': explosion if 'explosion' in locals() else None"
                content = re.sub(pattern, replacement, content, count=1)
                changes_made.append("Added explosion to return dict")

        # Step 2: Add explosion alert to final summary
        # Check if explosion alert already appears twice (once in analysis, once in summary)
        if content.count("🚀 主升段爆发行情侦测!") < 2:
            # Insert before the AI model accuracy display
            pattern = r"(\n\s+# 顯示AI模型準確度摘要)"

            if re.search(pattern, content):
                explosion_alert = '''
        # 爆发行情警告
        if result.get('explosion_detected', False):
            print("\\n" + "=" * 80)
            print("🚀 主升段爆发行情侦测!")
            print("=" * 80)
            print("📌 爆发行情特征:")
            print("   • 资金强势流入 (OBV > 20日均线)")
            print("   • 趋势加速 (10日均线斜率 > 30日均线斜率)")
            print("   • 处于周期初升段 (EARLY_UPCYCLE)")
            print("   • 量能放大 (量比 > 1.3x)")
            if result.get('explosion_data'):
                exp_data = result['explosion_data']
                print(f"\\n🔥 爆发行情数据:")
                print(f"   量比: {exp_data['volume_ratio']:.2f}x")
                print(f"   周期阶段: {exp_data['cycle_phase']}")
                print(f"   资金流入: {'是' if exp_data['money_inflow'] else '否'}")
                print(f"   趋势加速: {'是' if exp_data['trend_accelerating'] else '否'}")
            print("=" * 80)
'''
                content = re.sub(pattern, r'\1' + explosion_alert + '\n', content, count=1)
                changes_made.append("Added explosion alert to summary")

        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, ", ".join(changes_made) if changes_made else "Updated"
        else:
            return False, "No changes needed"

    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main function"""
    print("=" * 80)
    print("Adding explosion detection to final summary in all get_trading*.py files")
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
        success, message = add_explosion_to_summary(file_path)

        if success:
            print(f"[OK] {message}")
            success_count += 1
        else:
            if "No changes" in message:
                print(f"[SKIP] {message}")
                skip_count += 1
            else:
                print(f"[ERROR] {message}")
                error_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total files:    {len(files)}")
    print(f"Updated:        {success_count}")
    print(f"Already OK:     {skip_count}")
    print(f"Errors:         {error_count}")
    print("=" * 80)

    if success_count > 0:
        print(f"\n[OK] Successfully updated {success_count} files!")
    else:
        print("\n[INFO] All files already have explosion alert in summary")

if __name__ == "__main__":
    main()
