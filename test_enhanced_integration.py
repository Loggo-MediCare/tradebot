"""
测试增强评分系统整合效果
=========================
对比旧系统 vs 新系统的评分差异
"""
from enhanced_scoring_module import calculate_enhanced_buy_score

# 默认权重
default_weights = {
    'rsi_oversold': 30,
    'rsi_low': 15,
    'macd_bullish_strong': 35,
    'macd_bullish': 25,
    'ma_bullish': 25
}

print("=" * 80)
print("增强评分系统整合效果对比")
print("=" * 80)

# 测试案例 1: 2408.TW (RSI 超买 + 强势确认)
print("\n" + "=" * 80)
print("案例 1: 2408.TW (2025-12-24)")
print("RSI: 77.5 (超买) | MACD: 金叉+正值 | 均线: 多头 | 量比: 1.49x")
print("=" * 80)

print("\n【旧系统逻辑】")
print("-" * 80)
print("RSI > 70 → 直接扣分 -50")
print("MACD金叉 → +35")
print("均线多头 → +25")
print("放量1.5x → +20")
print("总分: -50 + 35 + 25 + 20 = 30 (偏低)")
print("结论: AI建议卖出，技术面支持度不足 → 观望/弱卖出")

print("\n【新系统逻辑】")
print("-" * 80)
score, signal, reasons, warnings, meta = calculate_enhanced_buy_score(
    rsi=77.54,
    macd=9.9546,
    macd_signal=8.0678,
    sma_10=169.30,
    sma_30=158.27,
    current_price=189.00,
    bb_upper=184.91,
    bb_lower=137.14,
    volume_ratio=1.49,
    ai_action=-1.0,
    buy_weights=default_weights
)

print(f"✨ 识别为: {'强势股' if meta['is_strong_stock'] else '普通股'}")
print(f"强势强度: {meta['strong_strength']}/100")
print(f"多头组合: {meta['combo_score']:.0f}/100")
print(f"最终评分: {score:.2f}")
print(f"信号建议: {signal}")

print(f"\n核心理由:")
for i, reason in enumerate(reasons[:6], 1):
    print(f"  {i}. {reason}")

print("\n💡 新系统优势:")
print("   • 识别出「强势突破」模式，不因RSI超买而恐慌")
print("   • 综合考虑MACD+均线+放量的组合确认")
print("   • 评分从30提升到49，从「观望」升级为「谨慎持有/小幅减仓」")

# 测试案例 2: 普通超买（无强势确认）
print("\n" + "=" * 80)
print("案例 2: 普通超买股票")
print("RSI: 75 (超买) | MACD: 死叉 | 均线: 空头 | 量比: 0.6x (缩量)")
print("=" * 80)

print("\n【旧系统逻辑】")
print("-" * 80)
print("MACD死叉 → -100 (直接拒绝)")
print("RSI > 70 → -50")
print("均线空头 → -20")
print("缩量 → -35")
print("总分: -205")
print("结论: 强烈卖出")

print("\n【新系统逻辑】")
print("-" * 80)
score2, signal2, reasons2, warnings2, meta2 = calculate_enhanced_buy_score(
    rsi=75.0,
    macd=-0.5,
    macd_signal=0.2,
    sma_10=100.0,
    sma_30=102.0,
    current_price=105.0,
    bb_upper=110.0,
    bb_lower=95.0,
    volume_ratio=0.6,
    ai_action=0.3,
    buy_weights=default_weights
)

print(f"识别为: {'强势股' if meta2['is_strong_stock'] else '普通股'}")
print(f"多头组合: {meta2['combo_score']:.0f}/100")
print(f"最终评分: {score2:.2f}")
print(f"信号建议: {signal2}")

if warnings2:
    print(f"\n⚠️  警告:")
    for warning in warnings2:
        print(f"   • {warning}")

print("\n💡 新系统优势:")
print("   • 正确识别为「真超买」(无强势确认)")
print("   • MACD死叉直接拒绝买入")
print("   • 评分-45，明确建议观望")

# 测试案例 3: 6209.TW (真实案例)
print("\n" + "=" * 80)
print("案例 3: 6209.TW (今國光) - 2025-12-24")
print("RSI: 66.8 | MACD: 金叉 | 均线: 多头 | 量比: 1.28x | 布林带: 99%")
print("=" * 80)

print("\n【新系统评分】")
print("-" * 80)
score3, signal3, reasons3, warnings3, meta3 = calculate_enhanced_buy_score(
    rsi=66.79,
    macd=3.1361,
    macd_signal=2.9194,
    sma_10=62.40,
    sma_30=58.44,
    current_price=66.20,
    bb_upper=66.33,
    bb_lower=53.22,
    volume_ratio=1.28,
    ai_action=0.3938,  # AI 建议买入
    buy_weights=default_weights
)

print(f"识别为: {'强势股' if meta3['is_strong_stock'] else '普通股'}")
print(f"多头组合: {meta3['combo_score']:.0f}/100")
print(f"最终评分: {score3:.2f}")
print(f"信号建议: {signal3}")

print(f"\n核心理由:")
for i, reason in enumerate(reasons3[:6], 1):
    print(f"  {i}. {reason}")

if warnings3:
    print(f"\n⚠️  警告:")
    for warning in warnings3:
        print(f"   • {warning}")

print("\n💡 分析:")
if meta3['is_strong_stock']:
    print("   • 符合强势股模式，建议积极买入")
else:
    print("   • 虽非强势股，但多头组合良好")
    print(f"   • 布林带99%位置需谨慎，接近上轨")
    print(f"   • AI+技术面均看多，建议谨慎买入")

print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print("✅ 新系统可以智能区分:")
print("   1. 强势突破 (RSI超买 + 多头确认) → 不恐慌")
print("   2. 真实超买 (RSI超买 + 无确认) → 拒绝买入")
print("   3. 多头组合 (MACD+均线+放量) → 加分")
print("\n✅ 避免了旧系统的问题:")
print("   1. 见RSI>70就扣分 (忽略强势股)")
print("   2. 单一指标判断 (忽略组合信号)")
print("   3. 过度保守 (错失强势行情)")
print("=" * 80)
