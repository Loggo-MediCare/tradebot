"""
批量更新所有交易信号文件
=========================
将增强评分系统整合到所有 get_trading_signal_*.py 文件
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import re
import shutil
from datetime import datetime

# 所有需要更新的文件
SIGNAL_FILES = [
    'get_trading_signal_1519.py',
    'get_trading_signal_2317.py',
    'get_trading_signal_2330.py',
    'get_trading_signal_2337.py',
    'get_trading_signal_2344.py',
    'get_trading_signal_2408.py',
    'get_trading_signal_3017.py',
    'get_trading_signal_3711.py',
    'get_trading_signal_3715.py',
    'get_trading_signal_4938.py',
    'get_trading_signal_6209.py',
    'get_trading_signal_6269.py',
    'get_trading_signal_6443.py',
    'get_trading_signal_6515.py',
    'get_trading_signal_6770.py',
    'get_trading_signal_6805.py',
    'get_trading_signal_8131.py',
    'get_trading_signal_8210.py',
    'get_trading_signal_aapl.py',
    'get_trading_signal_avgo.py',
    'get_trading_signal_goog.py',
    'get_trading_signal_mu.py',
    'get_trading_signal_nvda.py',
]

PROJECT_DIR = r'C:\Users\Silvi\Projects\trading-bot'

# 要添加的导入语句
IMPORT_STATEMENT = """
# 导入增强评分模块
from enhanced_scoring_module import calculate_enhanced_buy_score
"""

def backup_file(filepath):
    """备份原文件"""
    backup_dir = os.path.join(PROJECT_DIR, 'backups_before_enhanced_scoring')
    os.makedirs(backup_dir, exist_ok=True)

    filename = os.path.basename(filepath)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(backup_dir, f"{filename}.{timestamp}.bak")

    shutil.copy2(filepath, backup_path)
    return backup_path

def add_import(content):
    """添加导入语句"""
    # 查找已有的导入区域
    pattern = r'(from dynamic_signal_weights import DynamicWeightCalculator)'

    if 'from enhanced_scoring_module import' in content:
        print("  ⏭️  已存在增强模块导入")
        return content

    # 在 DynamicWeightCalculator 导入后添加
    if re.search(pattern, content):
        content = re.sub(
            pattern,
            r'\1' + IMPORT_STATEMENT,
            content
        )
        print("  ✅ 添加导入语句")
    else:
        print("  ⚠️  未找到导入位置，尝试在文件头部添加")
        # 在第一个 import 后添加
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                lines.insert(i + 1, IMPORT_STATEMENT)
                content = '\n'.join(lines)
                break

    return content

def update_buy_scoring_logic(content):
    """更新买入评分逻辑"""
    # 查找买入评分部分的开始
    # 寻找 "买入信号评分系统" 或类似的注释
    pattern = r'(# 🔥 买入信号评分系统.*?\n.*?buy_score = 0)'

    if not re.search(pattern, content, re.DOTALL):
        print("  ⚠️  未找到买入评分部分")
        return content, False

    # 替换为增强评分调用
    enhanced_code = """# 🔥 增强版买入信号评分系统
        buy_score, signal_override, buy_reasons, buy_warnings, buy_metadata = calculate_enhanced_buy_score(
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            sma_10=sma_10,
            sma_30=sma_30,
            current_price=current_price,
            bb_upper=bb_upper,
            bb_lower=bb_lower,
            volume_ratio=volume_ratio,
            ai_action=action_value,
            buy_weights=buy_weights
        )

        # 使用增强评分结果
        reasons = buy_reasons
        warnings = buy_warnings"""

    # 查找并替换整个买入评分区块
    # 从 "buy_score = 0" 到下一个主要的 if 语句之前
    buy_block_pattern = r'(# 🔥 买入信号评分系统.*?)(buy_score = 0.*?)(# 🔥 调整买入强度|# 如果评分过低)'

    if re.search(buy_block_pattern, content, re.DOTALL):
        content = re.sub(
            buy_block_pattern,
            r'\1' + enhanced_code + r'\n\n        \3',
            content,
            flags=re.DOTALL
        )
        print("  ✅ 更新买入评分逻辑")
        return content, True

    print("  ⚠️  买入评分区块模式不匹配")
    return content, False

def update_file(filename):
    """更新单个文件"""
    filepath = os.path.join(PROJECT_DIR, filename)

    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filename}")
        return False

    print(f"\n📝 处理: {filename}")

    # 备份
    backup_path = backup_file(filepath)
    print(f"  💾 备份: {os.path.basename(backup_path)}")

    # 读取文件
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  ❌ 读取失败: {e}")
        return False

    # 添加导入
    content = add_import(content)

    # 更新买入评分逻辑
    content, updated = update_buy_scoring_logic(content)

    if not updated:
        print(f"  ⚠️  评分逻辑未更新，保留原文件")
        # 可以选择是否保存（当前选择不保存）
        return False

    # 写回文件
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ 更新成功")
        return True
    except Exception as e:
        print(f"  ❌ 写入失败: {e}")
        # 恢复备份
        shutil.copy2(backup_path, filepath)
        print(f"  🔄 已恢复备份")
        return False

def main():
    print("=" * 80)
    print("批量更新交易信号文件 - 增强评分系统")
    print("=" * 80)
    print(f"\n总共 {len(SIGNAL_FILES)} 个文件需要更新\n")

    success_count = 0
    failed_files = []
    skipped_files = []

    for filename in SIGNAL_FILES:
        try:
            result = update_file(filename)
            if result:
                success_count += 1
            else:
                skipped_files.append(filename)
        except Exception as e:
            print(f"  ❌ 未预期的错误: {e}")
            failed_files.append(filename)

    # 总结报告
    print("\n" + "=" * 80)
    print("更新完成报告")
    print("=" * 80)
    print(f"✅ 成功更新: {success_count}/{len(SIGNAL_FILES)} 个文件")

    if skipped_files:
        print(f"\n⏭️  跳过的文件 ({len(skipped_files)} 个):")
        for f in skipped_files:
            print(f"   • {f}")

    if failed_files:
        print(f"\n❌ 失败的文件 ({len(failed_files)} 个):")
        for f in failed_files:
            print(f"   • {f}")

    backup_dir = os.path.join(PROJECT_DIR, 'backups_before_enhanced_scoring')
    print(f"\n💾 备份文件保存在: {backup_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
