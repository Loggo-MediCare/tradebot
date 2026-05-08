#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量更新脚本：为所有股票信号文件添加 FinBERT 情绪分析功能
"""

import os
import sys
import io
import re
import shutil
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================
# 配置：所有股票信号文件列表
# ==========================================
STOCK_CONFIGS = [
    # 美股 (9个)
    {'file': 'get_trading_signal_aapl.py', 'ticker': 'AAPL', 'name': 'Apple'},
    {'file': 'get_trading_signal_avgo.py', 'ticker': 'AVGO', 'name': 'Broadcom'},
    {'file': 'get_trading_signal_goog.py', 'ticker': 'GOOG', 'name': 'Google'},
    {'file': 'get_trading_signal_mu.py', 'ticker': 'MU', 'name': 'Micron'},
    {'file': 'get_trading_signal_nvda.py', 'ticker': 'NVDA', 'name': 'NVIDIA'},
    {'file': 'get_trading_signal_omer.py', 'ticker': 'OMER', 'name': 'Omeros'},
    {'file': 'get_trading_signal_alab.py', 'ticker': 'ALAB', 'name': 'Astera Labs'},
    {'file': 'get_trading_signal_nat.py', 'ticker': 'NAT', 'name': 'Nordic American Tankers'},
    {'file': 'get_trading_signal_htgc.py', 'ticker': 'HTGC', 'name': 'Hercules Capital'},

    # 欧股 (1个)
    {'file': 'get_trading_signal_rhm.py', 'ticker': 'RHM.DE', 'name': 'Rheinmetall'},

    # 台股 (26个)
    {'file': 'get_trading_signal_1519.py', 'ticker': '1519.TW', 'name': '华城'},
    {'file': 'get_trading_signal_2317.py', 'ticker': '2317.TW', 'name': '鸿海'},
    {'file': 'get_trading_signal_2330.py', 'ticker': '2330.TW', 'name': '台积电'},
    {'file': 'get_trading_signal_2337.py', 'ticker': '2337.TW', 'name': '旺宏'},
    {'file': 'get_trading_signal_2344.py', 'ticker': '2344.TW', 'name': '华邦电'},
    {'file': 'get_trading_signal_2360.py', 'ticker': '2360.TW', 'name': '致茂'},
    {'file': 'get_trading_signal_2408.py', 'ticker': '2408.TW', 'name': '南亚科'},
    {'file': 'get_trading_signal_2451.py', 'ticker': '2451.TW', 'name': '创见'},
    {'file': 'get_trading_signal_3017.py', 'ticker': '3017.TW', 'name': '奇鋐'},
    {'file': 'get_trading_signal_3653.py', 'ticker': '3653.TW', 'name': '健策'},
    {'file': 'get_trading_signal_3661.py', 'ticker': '3661.TW', 'name': '世芯-KY'},
    {'file': 'get_trading_signal_3711.py', 'ticker': '3711.TW', 'name': '日月光投控'},
    {'file': 'get_trading_signal_3715.py', 'ticker': '3715.TW', 'name': '定颖投控'},
    {'file': 'get_trading_signal_4938.py', 'ticker': '4938.TW', 'name': '和硕'},
    {'file': 'get_trading_signal_6209.py', 'ticker': '6209.TW', 'name': '今国光'},
    {'file': 'get_trading_signal_6269.py', 'ticker': '6269.TW', 'name': '台燿'},
    {'file': 'get_trading_signal_6442.py', 'ticker': '6442.TW', 'name': '兆丰金'},
    {'file': 'get_trading_signal_6443.py', 'ticker': '6443.TW', 'name': '元晶'},
    {'file': 'get_trading_signal_6515.py', 'ticker': '6515.TW', 'name': '颖霖'},
    {'file': 'get_trading_signal_6770.py', 'ticker': '6770.TW', 'name': '力积电'},
    {'file': 'get_trading_signal_6781.py', 'ticker': '6781.TW', 'name': 'AES-KY'},
    {'file': 'get_trading_signal_6805.py', 'ticker': '6805.TW', 'name': '富世达'},
    {'file': 'get_trading_signal_7769.py', 'ticker': '7769.TW', 'name': '霖扬'},
    {'file': 'get_trading_signal_8131.py', 'ticker': '8131.TW', 'name': '福懋科'},
    {'file': 'get_trading_signal_8110.py', 'ticker': '8110.TW', 'name': '华东'},
    {'file': 'get_trading_signal_8210.py', 'ticker': '8210.TW', 'name': '勤诚'},
]


def backup_file(file_path):
    """备份原文件"""
    if not os.path.exists(file_path):
        return False

    backup_dir = 'signal_backups'
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(backup_dir, f"{os.path.basename(file_path)}.{timestamp}.bak")

    shutil.copy2(file_path, backup_path)
    return True


def upgrade_signal_file(file_path, ticker):
    """
    升级单个信号文件，添加 FinBERT 支持

    Args:
        file_path: 信号文件路径
        ticker: 股票代码

    Returns:
        bool: 是否成功
    """
    if not os.path.exists(file_path):
        print(f"   ⚠️  文件不存在: {file_path}")
        return False

    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否已经升级
    if 'finbert_enhanced_scoring' in content:
        print(f"   ℹ️  已升级，跳过")
        return True

    # ==========================================
    # 修改 1: 更新导入语句
    # ==========================================
    old_import = 'from enhanced_scoring_module import calculate_enhanced_buy_score'
    new_import = (
        '# 导入增强评分模块（含FinBERT情绪分析）\n'
        'from finbert_enhanced_scoring import calculate_enhanced_buy_score_with_sentiment, format_sentiment_output'
    )

    if old_import in content:
        content = content.replace(old_import, new_import)
    else:
        # 如果找不到旧导入，在 dynamic_signal_weights 后面添加
        pattern = r'(from dynamic_signal_weights import DynamicWeightCalculator\n)'
        replacement = r'\1' + new_import + '\n'
        content = re.sub(pattern, replacement, content)

    # ==========================================
    # 修改 2: 添加 sentiment_result 占位变量
    # ==========================================
    pattern = r'(    # 8\. 生成交易建议\n)'
    replacement = r'    # 7.5 占位变量（将在买入信号部分填充）\n    sentiment_result = None\n\n\1'

    if '# 7.5 占位变量' not in content:
        content = re.sub(pattern, replacement, content)

    # ==========================================
    # 修改 3: 更新 calculate_enhanced_buy_score 调用
    # ==========================================
    # 查找并替换买入评分调用
    old_call_pattern = r'buy_score, signal_override, buy_reasons, buy_warnings, buy_metadata = calculate_enhanced_buy_score\('
    new_call = f"buy_score, signal_override, buy_reasons, buy_warnings, buy_metadata, sentiment_result = calculate_enhanced_buy_score_with_sentiment("

    content = re.sub(old_call_pattern, new_call, content)

    # 添加 symbol 参数（在 buy_weights 后面）
    # 匹配模式：buy_weights=buy_weights 后面可能有 ) 或 \n
    old_param_pattern = r'(buy_weights=buy_weights)(\s*\))'
    new_param = f"\\1,\n            symbol='{ticker}'\\2"

    content = re.sub(old_param_pattern, new_param, content)

    # ==========================================
    # 修改 4: 添加情绪分析输出
    # ==========================================
    # 在买入建议的操作建议后面添加情绪输出
    # 查找模式：print(f"      • 设置止损: ${current_price * 0.95:.2f} (-5%)")

    sentiment_output = '''
        # 显示FinBERT情绪分析结果
        if sentiment_result and sentiment_result['news_count'] > 0:
            print("\\n" + format_sentiment_output(sentiment_result))'''

    # 在最后一个止损建议后添加
    pattern = r'(            print\(f"      • 设置止损: \$\{current_price \* 0\.95:.2f\} \(-5%\)"\)\n)'

    # 检查是否已经添加
    if '# 显示FinBERT情绪分析结果' not in content:
        # 查找所有匹配位置
        matches = list(re.finditer(pattern, content))
        if matches:
            # 在最后一个匹配位置后添加
            last_match = matches[-1]
            insert_pos = last_match.end()
            content = content[:insert_pos] + sentiment_output + '\n' + content[insert_pos:]

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return True


def main():
    """主函数"""
    print("="*80)
    print("🚀 批量升级脚本：为所有股票信号文件添加 FinBERT 情绪分析")
    print("="*80)
    print(f"📊 待升级文件数量: {len(STOCK_CONFIGS)}")
    print(f"   • 美股: 9 个")
    print(f"   • 欧股: 1 个")
    print(f"   • 台股: 26 个")
    print("="*80)

    # 询问是否继续
    print("\n⚠️  警告：")
    print("   1. 此操作会修改所有 get_trading_signal_*.py 文件")
    print("   2. 原文件会自动备份到 signal_backups/ 目录")
    print("   3. 建议在升级前先提交 git 或手动备份\n")

    response = input("是否继续？(y/n): ").strip().lower()
    if response != 'y':
        print("\n❌ 操作已取消")
        return

    # 开始升级
    print("\n" + "="*80)
    print("📦 开始批量升级...")
    print("="*80)

    success_count = 0
    skip_count = 0
    fail_count = 0

    for idx, stock_config in enumerate(STOCK_CONFIGS, 1):
        file_name = stock_config['file']
        ticker = stock_config['ticker']
        name = stock_config['name']

        print(f"\n[{idx}/{len(STOCK_CONFIGS)}] {ticker} ({name})")
        print(f"   文件: {file_name}")

        file_path = os.path.join(os.getcwd(), file_name)

        # 备份文件
        if backup_file(file_path):
            print(f"   ✅ 已备份")
        else:
            print(f"   ⚠️  文件不存在，跳过")
            skip_count += 1
            continue

        # 升级文件
        try:
            if upgrade_signal_file(file_path, ticker):
                success_count += 1
                print(f"   ✅ 升级成功")
            else:
                fail_count += 1
                print(f"   ❌ 升级失败")
        except Exception as e:
            fail_count += 1
            print(f"   ❌ 升级失败: {e}")

    # 汇总报告
    print("\n" + "="*80)
    print("📊 批量升级完成")
    print("="*80)
    print(f"✅ 成功: {success_count} 个")
    print(f"⏭️  跳过: {skip_count} 个")
    print(f"❌ 失败: {fail_count} 个")
    print(f"📁 备份目录: ./signal_backups/")
    print("="*80)

    if success_count > 0:
        print("\n💡 后续步骤:")
        print("   1. 测试任意一个信号文件:")
        print("      python get_trading_signal_nvda.py")
        print("   2. 如果有问题，从 signal_backups/ 恢复")
        print("   3. 确认无误后，提交到 git")

    print("\n✅ 完成!")


if __name__ == '__main__':
    main()
