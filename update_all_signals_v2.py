"""
批量更新所有股票信号文件 - 完整版
包含买入和卖出的完整优化逻辑
"""

import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import shutil

# 需要更新的文件列表
signal_files = [
    'get_trading_signal_8131.py',
    'get_trading_signal_2330.py',
    'get_trading_signal_2344.py',
    'get_trading_signal_2317.py',
    'get_trading_signal_6770.py',
    'get_trading_signal_goog.py',
    'get_trading_signal_avgo.py',
    'get_trading_signal_2337.py',
    'get_trading_signal_mu.py',
    'get_trading_signal_aapl.py',
    'get_trading_signal_1519.py',
    'get_trading_signal_3017.py',
    'get_trading_signal_3711.py',
    'get_trading_signal_3715.py',
    'get_trading_signal_4938.py',
    'get_trading_signal_6209.py',
    'get_trading_signal_6269.py',
    'get_trading_signal_6443.py',
    'get_trading_signal_6515.py',
    'get_trading_signal_6805.py',
    'get_trading_signal_8210.py',
    'get_trading_signal_nvda.py',
]

def update_signal_file(filename):
    """直接复制2408的完整逻辑，只替换股票代码和模型路径"""

    source_file = r'C:\Users\Silvi\Projects\trading-bot\get_trading_signal_2408.py'
    target_file = os.path.join(r'C:\Users\Silvi\Projects\trading-bot', filename)

    if not os.path.exists(target_file):
        print(f"❌ 文件不存在: {filename}")
        return False

    try:
        # 读取2408的完整代码
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取股票代码和信息
        if filename.startswith('get_trading_signal_'):
            ticker_part = filename.replace('get_trading_signal_', '').replace('.py', '')

            # 判断是台股还是美股
            if ticker_part.isdigit():
                # 台股
                ticker_symbol = f"{ticker_part}.TW"
                # 特殊处理: 2317 使用 taiwan，其他使用 tw
                if ticker_part == '2317':
                    model_path = f"ppo_{ticker_part}_taiwan_improved.zip"
                else:
                    model_path = f"ppo_{ticker_part}_tw_improved.zip"
                stock_name_map = {
                    '2317': '鴻海',
                    '2330': '台積電',
                    '2337': '友達',
                    '2344': '華邦電',
                    '2408': '南亞科',
                    '6770': '力積電',
                    '8131': '台中銀',
                    '1519': '華城',
                    '3017': '奇鋐',
                    '3711': '日月光投控',
                    '3715': '定穎投控',
                    '4938': '和碩',
                    '6209': '今國光',
                    '6269': '台郡',
                    '6443': '元晶',
                    '6515': '穎崴',
                    '6805': '富世達',
                    '8210': '勤誠',
                }
                stock_name = stock_name_map.get(ticker_part, f'台股{ticker_part}')
                currency = 'NT$'
            else:
                # 美股
                ticker_symbol = ticker_part.upper()
                model_path = f"ppo_{ticker_part.lower()}_improved.zip"
                stock_name_map = {
                    'aapl': 'Apple',
                    'goog': 'Google',
                    'avgo': 'Broadcom',
                    'mu': 'Micron',
                    'nvda': 'NVIDIA',
                }
                stock_name = stock_name_map.get(ticker_part.lower(), ticker_symbol)
                currency = '$'

            # 替换内容 (顺序很重要！先替换长字符串，再替换短字符串)
            content = content.replace('ppo_2408_tw_improved.zip', model_path)  # 先替换模型路径
            content = content.replace('(南亞科技)', f'({stock_name})')  # 再替换带括号的名称
            content = content.replace('2408.TW', ticker_symbol)  # 再替换股票代码
            content = content.replace('南亞科', stock_name)  # 再替换股票名称
            content = content.replace('2408', ticker_part)  # 最后替换纯数字（避免误替换）

            # 替换货币符号
            if currency == '$':
                content = content.replace('NT$', '$')

        # 写入目标文件
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✅ 更新成功: {filename}")
        return True

    except Exception as e:
        print(f"❌ 更新失败 {filename}: {e}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("批量更新股票信号文件 - 完整版 V2")
    print("=" * 70)
    print(f"\n将更新 {len(signal_files)} 个文件...")
    print("\n完整改进内容:")
    print("  买入逻辑:")
    print("    1. 成交量评分系统")
    print("    2. 缩量(<0.7x)扣35分")
    print("    3. 评分<20改为'观望'")
    print("    4. 🆕 基本面分析: 优质公司超跌+50分")
    print("    5. 🆕 大型公司加成: 营收>35B额外+20分")
    print("  卖出逻辑:")
    print("    6. 极度放量(>2.5x)+RSI<80 = 100%持有")
    print("    7. 强势股评分打2折")
    print("    8. 基于回测数据的多因子判断")
    print("    9. 🆕 基本面保护: 优质公司超跌=不卖出")
    print("\n" + "=" * 70)
    print("\n开始更新...\n")

    success_count = 0
    for filename in signal_files:
        if update_signal_file(filename):
            success_count += 1

    print("\n" + "=" * 70)
    print(f"完成! 成功更新 {success_count}/{len(signal_files)} 个文件")
    print("=" * 70)
