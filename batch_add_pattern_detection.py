"""
批次更新腳本: 為所有 get_trading_signal_*.py 文件添加 pattern detection 功能
================================================================================
添加以下模組整合:
1. triangle_pattern - 三角收斂型態
2. breakout_detector - 真假突破檢測
3. pattern_engine - 圖表型態識別
4. volume_surge_detector - 爆量信號檢測
"""

import os
import re
import glob

# 需要添加的 imports
NEW_IMPORTS = """from triangle_pattern import detect_triangle, triangle_breakout
from breakout_detector import get_breakout_signal
from pattern_engine import get_pattern_signal
from volume_surge_detector import get_volume_signal
"""

# 需要添加的 pattern detection 代碼塊 (買入信號部分)
PATTERN_DETECTION_CODE = '''
        # 三角收斂型態檢測
        if detect_triangle(df):
            status = triangle_breakout(df)
            if status == "BREAK_UP":
                buy_score += 10
                buy_reasons.append("三角收斂向上突破")
            elif status == "BREAK_DOWN":
                buy_warnings.append("跌破三角收斂")

        # 真假突破檢測
        breakout_signal = get_breakout_signal(df)
        if breakout_signal['detected']:
            if breakout_signal['type'] == 'TRUE_BREAKOUT':
                buy_score += 15
                buy_reasons.append(breakout_signal['signal_text'])
            elif breakout_signal['type'] == 'FALSE_BREAKOUT':
                buy_score -= 10
                buy_warnings.append(breakout_signal['signal_text'])

        # 圖表型態識別
        pattern_signal = get_pattern_signal(df)
        if pattern_signal['patterns']:
            if pattern_signal['score_adjustment'] > 0:
                buy_score += pattern_signal['score_adjustment']
                buy_reasons.append(f"型態: {pattern_signal['signal_text']}")
            elif pattern_signal['score_adjustment'] < 0:
                buy_warnings.append(f"型態警示: {pattern_signal['signal_text']}")

        # 爆量信號檢測 (法人上車)
        volume_signal = get_volume_signal(df)
        if volume_signal['surge'] and volume_signal['surge']['detected']:
            if volume_signal['surge']['type'] == 'SURGE_UP':
                buy_score += 15
                buy_reasons.append(volume_signal['surge']['signal_text'])
            elif volume_signal['surge']['type'] == 'SURGE_DOWN':
                buy_warnings.append(volume_signal['surge']['signal_text'])

'''

def has_pattern_imports(content):
    """檢查是否已有 pattern imports"""
    return 'from pattern_engine import' in content or 'from breakout_detector import' in content

def has_pattern_detection_code(content):
    """檢查是否已有 pattern detection 代碼"""
    return 'get_breakout_signal(df)' in content or 'get_pattern_signal(df)' in content

def add_imports(content):
    """添加 imports 到文件"""
    # 找到最後一個 from ... import 語句
    import_pattern = r'(from \w+[\w_]* import [^\n]+\n)(?=\n|\s*#|\s*class|\s*def)'

    # 找到所有 import 語句
    matches = list(re.finditer(r'^(from .+ import .+|import .+)$', content, re.MULTILINE))

    if matches:
        last_import = matches[-1]
        insert_pos = last_import.end()

        # 確保在正確位置插入 (在最後一個 import 後)
        new_content = content[:insert_pos] + '\n' + NEW_IMPORTS + content[insert_pos:]
        return new_content

    return content

def add_pattern_detection(content):
    """添加 pattern detection 代碼到買入信號部分"""

    # 尋找插入點: 在 buy_score 計算之後, buy_score = max(0, min(100, buy_score)) 之前
    # 或者在 "加入MA50斜率評分調整" 之後

    # 方案1: 找到 "buy_score = max(0, min(100, buy_score))" 並在之前插入
    pattern1 = r'(\s+buy_score = max\(0, min\(100, buy_score\)\))'
    match1 = re.search(pattern1, content)

    if match1:
        insert_pos = match1.start()
        new_content = content[:insert_pos] + PATTERN_DETECTION_CODE + content[insert_pos:]
        return new_content

    # 方案2: 找到 "ma50_slope_adjustment" 相關代碼後插入
    pattern2 = r'(ma50_slope_adjustment.*?buy_warnings\.append.*?MA50.*?\n)'
    match2 = re.search(pattern2, content, re.DOTALL)

    if match2:
        insert_pos = match2.end()
        new_content = content[:insert_pos] + PATTERN_DETECTION_CODE + content[insert_pos:]
        return new_content

    # 方案3: 找到 "buy_reasons.append" 或 "buy_warnings.append" 附近插入
    pattern3 = r'(if ma50_slope_adjustment [<>] 0:.*?buy_warnings\.append.*?\n)'
    match3 = re.search(pattern3, content, re.DOTALL)

    if match3:
        insert_pos = match3.end()
        new_content = content[:insert_pos] + PATTERN_DETECTION_CODE + content[insert_pos:]
        return new_content

    return None  # 無法找到插入點

def process_file(filepath):
    """處理單個文件"""
    print(f"\n處理: {os.path.basename(filepath)}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  [ERROR] 讀取失敗: {e}")
        return False

    modified = False

    # 1. 檢查並添加 imports
    if not has_pattern_imports(content):
        content = add_imports(content)
        print("  [OK] 已添加 pattern imports")
        modified = True
    else:
        print("  [SKIP] 已有 pattern imports")

    # 2. 檢查並添加 pattern detection 代碼
    if not has_pattern_detection_code(content):
        new_content = add_pattern_detection(content)
        if new_content:
            content = new_content
            print("  [OK] 已添加 pattern detection 代碼")
            modified = True
        else:
            print("  [WARN] 無法找到插入點，請手動添加")
    else:
        print("  [SKIP] 已有 pattern detection 代碼")

    # 3. 寫回文件
    if modified:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  [SAVED] 已保存更新")
            return True
        except Exception as e:
            print(f"  [ERROR] 保存失敗: {e}")
            return False

    return False

def main():
    print("=" * 70)
    print("Batch Add Pattern Detection to Trading Signal Files")
    print("=" * 70)

    # 獲取所有 get_trading_signal_*.py 文件
    pattern = os.path.join(os.getcwd(), "get_trading_signal_*.py")
    files = glob.glob(pattern)

    # 排除備份文件
    files = [f for f in files if '_bak' not in f and 'test' not in f.lower()]

    print(f"\nFound {len(files)} signal files")

    updated_count = 0
    skipped_count = 0
    failed_count = 0

    for filepath in sorted(files):
        result = process_file(filepath)
        if result:
            updated_count += 1
        elif result is False:
            skipped_count += 1

    print("\n" + "=" * 70)
    print(f"Completed! Updated: {updated_count}, Skipped: {skipped_count}")
    print("=" * 70)

if __name__ == "__main__":
    main()
