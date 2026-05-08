"""
批量更新所有股票信号文件的卖出策略
将2408的改进卖出逻辑应用到所有信号文件
"""

import os
import re

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

# 新的成交量计算代码（添加到解析交易信号部分）
volume_code = '''    current_volume = float(latest_data['volume'])

    # 计算平均成交量（过去20天）
    avg_volume_20 = float(df['volume'].tail(20).mean())

    # 计算成交量比率
    volume_ratio = (current_volume / avg_volume_20) if avg_volume_20 > 0 else 1.0
'''

# 新的技术指标显示（添加成交量信息）
volume_display = '''    print(f"成交量:          {int(current_volume):,}")
    print(f"20日平均量:      {int(avg_volume_20):,}  {'[放量]' if volume_ratio > 1.5 else '[缩量]' if volume_ratio < 0.7 else '[正常]'}")
    print(f"量比:            {volume_ratio:.2f}x")
'''

# 改进的卖出逻辑
improved_sell_logic = '''        # 🔥 改进的卖出判断逻辑
        # 计算更多技术指标
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
        is_macd_bearish = macd < macd_signal
        is_trending_down = sma_10 < sma_30

        # 卖出信号评分系统（0-100分）
        sell_score = 0
        reasons = []

        # 1. RSI 超买判断（严格化：只在 RSI > 70 时才算）
        if rsi > 80:
            sell_score += 40
            reasons.append(f"RSI 严重超买 ({rsi:.1f} > 80)")
        elif rsi > 70:
            sell_score += 25
            reasons.append(f"RSI 超买 ({rsi:.1f} > 70)")
        elif rsi > 65:
            sell_score += 10
            reasons.append(f"RSI 偏高 ({rsi:.1f})")

        # 2. MACD 死叉（重要信号）
        if is_macd_bearish:
            sell_score += 30
            reasons.append("MACD 死叉,趋势转弱")

        # 3. 均线排列（空头排列）
        if is_trending_down:
            sell_score += 20
            reasons.append("短期均线下穿长期均线")

        # 4. 布林带位置（接近上轨才算）
        if bb_position > 90:
            sell_score += 25
            reasons.append(f"价格接近布林带上轨 ({bb_position:.1f}%)")
        elif bb_position > 80:
            sell_score += 10
            reasons.append(f"价格偏高,接近布林带上轨")

        # 5. 价格远高于均线（回调风险）
        price_vs_sma10 = ((current_price - sma_10) / sma_10) * 100
        if price_vs_sma10 > 10:
            sell_score += 15
            reasons.append(f"价格远高于10日均线 (+{price_vs_sma10:.1f}%)")
        elif price_vs_sma10 > 5:
            sell_score += 5
            reasons.append(f"价格高于10日均线 (+{price_vs_sma10:.1f}%)")

        # 6. 成交量分析（量价配合）
        # 🔥 新增：强势股判断（趋势强于指标）
        is_strong_trend = (not is_macd_bearish) and (not is_trending_down) and (volume_ratio > 1.2)

        if is_strong_trend and rsi > 70:
            # ✅ 强势股特征：高RSI + 多头趋势 + 放量 = 不应卖出！
            # 大幅降低卖出评分
            sell_score = int(sell_score * 0.3)  # 评分打3折
            reasons.clear()  # 清除之前的卖出理由
            reasons.append(f"⚠️  虽然 RSI 超买,但趋势强劲")
            reasons.append(f"MACD 金叉 + 均线多头 + 放量 (量比 {volume_ratio:.1f}x)")
            reasons.append(f"可能是强势突破,建议继续持有或小幅减仓")
        elif volume_ratio > 2.0 and rsi > 70:
            # 高位放量 + 超买 = 强烈卖出信号（可能是出货）
            sell_score += 20
            reasons.append(f"高位放量 (量比 {volume_ratio:.1f}x)，疑似出货")
        elif volume_ratio > 1.5 and rsi > 70:
            # 适度放量 + 超买（但不是强势股）
            sell_score += 10
            reasons.append(f"超买区放量 (量比 {volume_ratio:.1f}x)")
        elif volume_ratio < 0.5 and current_price > sma_10:
            # 价涨量缩 = 上涨乏力
            sell_score += 15
            reasons.append(f"价涨量缩 (量比 {volume_ratio:.1f}x)，上涨乏力")

        # 调整卖出强度和建议比例
        adjusted_strength = min(sell_score / 100, 1.0)  # 根据评分调整强度
        suggested_sell_ratio = int(adjusted_strength * 100)

        print(f"\\n{signal_emoji} 信号: {signal}")
        print(f"   AI 模型强度: {strength:.2f} / 1.00")
        print(f"   技术指标评分: {sell_score} / 100")
        print(f"   综合建议强度: {adjusted_strength:.2f}")
        print(f"   建议卖出比例: {suggested_sell_ratio}%")
        print(f"   建议卖出价格区间: ${suggested_price_low:.2f} - ${suggested_price_high:.2f}")

        if reasons:
            print(f"\\n   📌 卖出理由:")
            for i, reason in enumerate(reasons, 1):
                print(f"      {i}. {reason}")

        # 根据评分给出不同的操作建议
        print(f"\\n   💡 操作建议:")
        if sell_score >= 70:
            print(f"      •  多个卖出信号确认,建议尽快卖出")
            print(f"      • 可分2-3批卖出,保留少量仓位")
            print(f"      • 设置止损: ${current_price * 0.97:.2f} (-3%)")
        elif sell_score >= 50:
            print(f"      • 适度卖出,建议卖出 {suggested_sell_ratio}% 仓位")
            print(f"      • 保留部分仓位观察后续走势")
            print(f"      • 如果 RSI 继续上升,再卖出剩余仓位")
        else:
            print(f"      • AI 建议卖出,但技术指标支持度较弱")
            print(f"      • 可考虑小幅减仓 20-30%")
            print(f"      • 密切关注 MACD 和 RSI 变化")
'''


def update_signal_file(filename):
    """更新单个信号文件"""
    filepath = os.path.join(r'C:\Users\Silvi\Projects\trading-bot', filename)

    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filename}")
        return False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. 添加成交量计算（在 bb_lower 之后）
        if 'current_volume' not in content:
            content = re.sub(
                r"(bb_lower = float\(latest_data\['bb_lower'\]\))",
                r"\1\n" + volume_code,
                content
            )

        # 2. 添加成交量显示（在布林带位置之后）
        if '量比:' not in content:
            content = re.sub(
                r'(print\(f"当前价格位置:.*?\(布林带内\)"\))',
                r'\1\n' + volume_display,
                content
            )

        # 3. 替换旧的卖出逻辑
        # 找到 elif action_value < -0.1: 之后的内容
        # 替换从 print(f"\n{signal_emoji} 信号: {signal}") 到 else: 之前的所有内容

        # 使用更精确的正则表达式
        pattern = r'(elif action_value < -0\.1:.*?suggested_price_high = current_price \* 1\.005\s*\n)(.*?)(    else:)'

        if re.search(pattern, content, re.DOTALL):
            content = re.sub(
                pattern,
                r'\1' + improved_sell_logic + '\n\n\3',
                content,
                flags=re.DOTALL
            )

        # 写回文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✅ 更新成功: {filename}")
        return True

    except Exception as e:
        print(f"❌ 更新失败 {filename}: {e}")
        return False


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 70)
    print("批量更新股票信号文件的卖出策略")
    print("=" * 70)
    print(f"\n将更新 {len(signal_files)} 个文件...")
    print("\n改进内容:")
    print("  1. 添加成交量分析")
    print("  2. 严格化 RSI 判断 (>70 才算超买)")
    print("  3. 评分系统 (0-100分)")
    print("  4. 强势股识别 (趋势强于指标)")
    print("  5. 动态卖出比例建议")
    print("\n" + "=" * 70)

    success_count = 0
    for filename in signal_files:
        if update_signal_file(filename):
            success_count += 1

    print("\n" + "=" * 70)
    print(f"完成! 成功更新 {success_count}/{len(signal_files)} 个文件")
    print("=" * 70)
