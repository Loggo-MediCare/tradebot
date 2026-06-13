"""
Batch Update Script: Add Advanced Breakout Detection
=====================================================
This script updates all trading signal generators to include
the advanced breakout pattern detection (W底/杯柄/旗形/三角).
"""

import os
import re
import glob

def update_signal_file(filepath):
    """Update a single signal generator file with advanced breakout detection"""

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip if already updated
    if 'get_advanced_breakout_signal' in content:
        print(f"  [SKIP] Already updated: {os.path.basename(filepath)}")
        return False

    # Skip if doesn't have the breakout_detector import
    if 'from breakout_detector import get_breakout_signal' not in content:
        print(f"  [SKIP] No breakout_detector import: {os.path.basename(filepath)}")
        return False

    # Skip if doesn't have structure pattern analysis (different template)
    if '結構型態分析' not in content:
        print(f"  [SKIP] Different template: {os.path.basename(filepath)}")
        return False

    modified = False

    # 1. Update import statement
    old_import = 'from breakout_detector import get_breakout_signal'
    new_import = 'from breakout_detector import get_breakout_signal, get_advanced_breakout_signal, format_advanced_breakout_output'

    if old_import in content and new_import not in content:
        content = content.replace(old_import, new_import)
        modified = True

    # 2. Add advanced breakout section after structure pattern analysis
    # Find the pattern: structure_score_bonus = 0 followed by # 8. 生成交易建议

    old_section = '''    except Exception as e:
        print(f"   ⚠️  結構型態分析失敗: {e}")
        structure_score_bonus = 0

    # 8. 生成交易建议'''

    new_section = '''    except Exception as e:
        print(f"   ⚠️  結構型態分析失敗: {e}")
        structure_score_bonus = 0

    # 7.9 進階突破型態檢測 (W底/杯柄/旗形/三角)
    print("\\n" + "=" * 80)
    print("🚀 進階突破型態檢測")
    print("=" * 80)

    advanced_breakout_score = 0
    advanced_breakout_patterns = []
    try:
        advanced_signal = get_advanced_breakout_signal(df)
        if advanced_signal['detected']:
            print(f"✅ 檢測到 {len(advanced_signal['patterns'])} 個突破型態:")
            print(format_advanced_breakout_output(advanced_signal))
            advanced_breakout_score = advanced_signal['total_score']
            advanced_breakout_patterns = advanced_signal['patterns']
        else:
            print("   未檢測到進階突破型態")
    except Exception as e:
        print(f"   ⚠️  進階突破檢測失敗: {e}")

    # 8. 生成交易建议'''

    if old_section in content:
        content = content.replace(old_section, new_section)
        modified = True

    # 3. Add score integration in buy logic
    old_buy_logic = '''        # 加入結構型態評分調整 (W底/鍋底)
        if structure_score_bonus > 0:
            buy_score += structure_score_bonus
            buy_reasons.append(f"結構型態加分 (+{structure_score_bonus}分)")

        buy_score = max(0, min(100, buy_score))'''

    new_buy_logic = '''        # 加入結構型態評分調整 (W底/鍋底)
        if structure_score_bonus > 0:
            buy_score += structure_score_bonus
            buy_reasons.append(f"結構型態加分 (+{structure_score_bonus}分)")

        # 加入進階突破型態評分 (W底/杯柄/旗形/三角)
        if advanced_breakout_score > 0:
            buy_score += advanced_breakout_score
            for pattern in advanced_breakout_patterns:
                signal_text = pattern.get('signal_text', '')
                if signal_text:
                    buy_reasons.append(f"🚀 {signal_text}")

        buy_score = max(0, min(100, buy_score))'''

    if old_buy_logic in content and new_buy_logic not in content:
        content = content.replace(old_buy_logic, new_buy_logic)
        modified = True

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  [OK] Updated: {os.path.basename(filepath)}")
        return True
    else:
        print(f"  [SKIP] No changes needed: {os.path.basename(filepath)}")
        return False

def main():
    print("=" * 70)
    print("Batch Update: Advanced Breakout Detection")
    print("=" * 70)

    # Find all signal generator files
    base_path = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(base_path, 'get_trading_signal_*.py')
    files = glob.glob(pattern)

    print(f"\nFound {len(files)} signal generator files")
    print("-" * 70)

    updated_count = 0
    skipped_count = 0

    for filepath in sorted(files):
        # Skip special files
        basename = os.path.basename(filepath)
        if any(x in basename for x in ['gemini', 'aiwan', 'rhm_de', '3491o', '3110t']):
            print(f"  [SKIP] Special file: {basename}")
            skipped_count += 1
            continue

        if update_signal_file(filepath):
            updated_count += 1
        else:
            skipped_count += 1

    print("\n" + "=" * 70)
    print(f"Batch Update Complete!")
    print(f"  Updated: {updated_count} files")
    print(f"  Skipped: {skipped_count} files")
    print("=" * 70)

if __name__ == "__main__":
    main()
