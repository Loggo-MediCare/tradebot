"""
批量更新所有交易信号文件 - 添加动态权重系统
基于特征重要性自适应调整评分权重
"""
import os
import re
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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
    'get_trading_signal_4991.py',
    'get_trading_signal_6175.py',
    'get_trading_signal_6209.py',
    'get_trading_signal_6269.py',
    'get_trading_signal_6443.py',
    'get_trading_signal_6515.py',
    'get_trading_signal_6805.py',
    'get_trading_signal_8210.py',
    'get_trading_signal_nvda.py',
]

def update_signal_file(filename):
    """为信号文件添加动态权重支持"""
    filepath = os.path.join(r'C:\Users\Silvi\Projects\trading-bot', filename)

    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filename}")
        return False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        modified = False

        # 1. 添加 DynamicWeightCalculator 导入
        if 'from dynamic_signal_weights import DynamicWeightCalculator' not in content:
            # 在 warnings 导入后添加
            pattern = r"(import warnings\nwarnings\.filterwarnings\('ignore'\))\n"
            replacement = r"\1\n\n# 导入动态权重计算器\nfrom dynamic_signal_weights import DynamicWeightCalculator\n"

            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                modified = True
            else:
                print(f"⚠️  未找到导入插入点: {filename}")
                return False

        # 2. 添加权重计算器初始化
        if 'weight_calc = DynamicWeightCalculator' not in content:
            # 提取股票代码
            ticker_match = re.search(r"yf\.download\('([^']+)'", content)
            if not ticker_match:
                print(f"⚠️  无法提取股票代码: {filename}")
                return False

            ticker = ticker_match.group(1)

            # 在技术指标分析后、AI交易信号前添加
            pattern = r'(    print\(f"量比:\s+\{volume_ratio:.2f\}x"\)\s*\n)'
            replacement = f'''\1
    # 7. 初始化动态权重计算器
    weight_calc = DynamicWeightCalculator('{ticker}')
    buy_weights = weight_calc.get_buy_weights()
    sell_weights = weight_calc.get_sell_weights()

'''
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                modified = True

                # 更新后续注释编号
                content = content.replace('# 7. 生成交易建议', '# 8. 生成交易建议')
                content = content.replace('# 8. 风险提示', '# 9. 风险提示')
            else:
                print(f"⚠️  未找到权重初始化插入点: {filename}")

        # 3. 替换买入信号评分权重
        buy_replacements = [
            # RSI 超卖
            (r'buy_score \+= 25\s*\n\s*reasons\.append\(f"RSI超卖',
             r'buy_score += buy_weights[\'rsi_oversold\']\n                reasons.append(f"RSI超卖'),

            # RSI 偏低
            (r'buy_score \+= 15\s*\n\s*reasons\.append\(f"RSI偏低',
             r'buy_score += buy_weights[\'rsi_low\']\n                reasons.append(f"RSI偏低'),

            # MACD 金叉且为正值
            (r'buy_score \+= 25\s*\n\s*reasons\.append\("MACD金叉且为正值"\)',
             r'buy_score += buy_weights[\'macd_bullish_strong\']\n                reasons.append("MACD金叉且为正值")'),

            # MACD 金叉
            (r'elif macd > macd_signal:\s*\n\s*buy_score \+= 15\s*\n\s*reasons\.append\("MACD金叉"\)',
             r'elif macd > macd_signal:\n                buy_score += buy_weights[\'macd_bullish\']\n                reasons.append("MACD金叉")'),

            # 均线多头排列
            (r'buy_score \+= 15\s*\n\s*reasons\.append\("均线多头排列"\)',
             r'buy_score += buy_weights[\'ma_bullish\']\n                reasons.append("均线多头排列")'),

            # 价格低于短期均线
            (r'buy_score \+= 10\s*\n\s*reasons\.append\("价格低于短期均线"\)',
             r'buy_score += buy_weights[\'price_below_ma\']\n                reasons.append("价格低于短期均线")'),
        ]

        for pattern, replacement in buy_replacements:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                modified = True

        # 4. 替换卖出信号评分权重
        sell_replacements = [
            # RSI 严重超买
            (r'sell_score \+= 40\s*\n\s*reasons\.append\(f"RSI 严重超买',
             r'sell_score += sell_weights[\'rsi_severe\']\n            reasons.append(f"RSI 严重超买'),

            # RSI 超买
            (r'elif rsi > 70:\s*\n\s*sell_score \+= 25\s*\n\s*reasons\.append\(f"RSI 超买',
             r'elif rsi > 70:\n            sell_score += sell_weights[\'rsi_high\']\n            reasons.append(f"RSI 超买'),

            # RSI 偏高
            (r'elif rsi > 65:\s*\n\s*sell_score \+= 10\s*\n\s*reasons\.append\(f"RSI 偏高',
             r'elif rsi > 65:\n            sell_score += sell_weights[\'rsi_mild\']\n            reasons.append(f"RSI 偏高'),

            # MACD 死叉
            (r'sell_score \+= 30\s*\n\s*reasons\.append\("MACD 死叉',
             r'sell_score += sell_weights[\'macd_bearish\']\n            reasons.append("MACD 死叉'),

            # 均线空头排列
            (r'sell_score \+= 20\s*\n\s*reasons\.append\("短期均线下穿长期均线"\)',
             r'sell_score += sell_weights[\'ma_bearish\']\n            reasons.append("短期均线下穿长期均线")'),

            # 布林带上轨
            (r'sell_score \+= 25\s*\n\s*reasons\.append\(f"价格接近布林带上轨',
             r'sell_score += sell_weights[\'bb_upper\']\n            reasons.append(f"价格接近布林带上轨'),

            # 布林带偏高
            (r'elif bb_position > 80:\s*\n\s*sell_score \+= 10\s*\n\s*reasons\.append\(f"价格偏高',
             r'elif bb_position > 80:\n            sell_score += sell_weights[\'bb_high\']\n            reasons.append(f"价格偏高'),

            # 价格远高于均线
            (r'sell_score \+= 15\s*\n\s*reasons\.append\(f"价格远高于10日均线',
             r'sell_score += sell_weights[\'price_vs_ma_high\']\n            reasons.append(f"价格远高于10日均线'),

            # 价格高于均线
            (r'elif price_vs_sma10 > 5:\s*\n\s*sell_score \+= 5\s*\n\s*reasons\.append\(f"价格高于10日均线',
             r'elif price_vs_sma10 > 5:\n            sell_score += sell_weights[\'price_vs_ma_mild\']\n            reasons.append(f"价格高于10日均线'),
        ]

        for pattern, replacement in sell_replacements:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                modified = True

        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 更新成功: {filename}")
            return True
        else:
            print(f"⏭️  跳过 (已是最新): {filename}")
            return True

    except Exception as e:
        print(f"❌ 更新失败 {filename}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("批量更新交易信号文件 - 添加动态权重系统")
    print("=" * 70)
    print(f"\n将更新 {len(signal_files)} 个文件...")
    print("\n新增功能:")
    print("  1. ✅ 导入 DynamicWeightCalculator")
    print("  2. ✅ 根据特征重要性计算动态权重")
    print("  3. ✅ 买入信号使用自适应评分")
    print("  4. ✅ 卖出信号使用自适应评分")
    print("  5. ✅ 无特征数据时自动回退到默认权重")
    print("\n智能权重分配:")
    print("  • RSI重要性高(如NVDA 5%) → RSI权重提升")
    print("  • RSI重要性低(如2330 3%) → RSI权重降低")
    print("  • MA重要性高(如2330 26%) → 均线权重大幅提升")
    print("  • MACD/BB根据实际重要性动态调整")
    print("\n" + "=" * 70)
    print("\n开始更新...\n")

    success_count = 0
    for filename in signal_files:
        if update_signal_file(filename):
            success_count += 1

    print("\n" + "=" * 70)
    print(f"完成! 成功更新 {success_count}/{len(signal_files)} 个文件")
    print("=" * 70)
    print("\n📝 下一步:")
    print("  1. 运行各股票的训练文件生成 JSON 特征重要性数据")
    print("  2. 测试交易信号，观察权重是否按特征重要性调整")
    print("  3. 对比同一股票使用动态权重前后的信号差异")
    print("=" * 70)
