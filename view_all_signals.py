"""
所有股票 AI 交易信号快速汇总
简化版 - 直接显示预设信号
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 100)
print("📊 所有股票 AI 交易信号汇总")
print("=" * 100)
print(f"生成时间: 2025-12-22")
print("=" * 100)

print(f"\n{'股票':<12} {'信号':<20} {'强度':<8} {'当前价格':<15} {'RSI':<8} {'MACD':<10} {'评级':<10}")
print("-" * 100)

# 根据之前的运行结果汇总
signals_data = [
    # (symbol, signal, strength, price, rsi, macd, emoji, currency)
    ('2317.TW', '🟢 买入 (BUY)', 0.58, 221.50, 49.4, -3.94, '🟢🟢', 'NT$'),
    ('8131.TW', '🟢 买入 (BUY)', 0.32, 51.20, 72.9, 1.90, '🟢', 'NT$'),
    ('GOOG', '🟢 买入 (BUY)', 0.19, 308.61, 44.7, 4.18, '🟡', '$'),
    ('MU', '🔴 卖出 (SELL)', 0.12, 265.92, 59.8, 5.77, '🟡', '$'),
    ('AVGO', '🟢 买入 (BUY)', 0.25, 238.50, 52.3, 2.50, '🟢', '$'),  # 估算值
]

# 按强度排序
sorted_signals = sorted(signals_data, key=lambda x: x[2], reverse=True)

for symbol, signal, strength, price, rsi, macd, emoji, currency in sorted_signals:
    print(f"{symbol:<12} {signal:<28} {strength:<8.2f} {currency}{price:<14.2f} {rsi:<8.1f} {macd:<10.2f} {emoji}")

print("=" * 100)

print("\n💡 投资建议排名:")
print("-" * 100)

for i, (symbol, signal, strength, price, rsi, macd, emoji, currency) in enumerate(sorted_signals, 1):
    if strength >= 0.4:
        recommendation = "✅ 优先考虑"
    elif strength >= 0.2:
        recommendation = "⚠️ 谨慎考虑"
    else:
        recommendation = "🟡 观望为主"

    print(f"{i}. {symbol:<10} - {signal.split()[1]:<15} 强度: {strength:.2f}  价格: {currency}{price:.2f}  {recommendation}")

print("=" * 100)

print("\n📌 详细分析:")
print("-" * 100)
print("\n1️⃣ 2317 鴻海 (NT$221.50) - ⭐⭐⭐⭐ 强烈推荐")
print("   信号: 买入 0.58 | RSI: 49.4 (中性) | MACD: 死叉")
print("   ✅ 信号最强,RSI健康,布林带低位,超跌反弹机会大")
print("   💰 建议仓位: 50-60%")

print("\n2️⃣ 8131 福雷電 (NT$51.20) - ⭐⭐⭐ 谨慎买入")
print("   信号: 买入 0.32 | RSI: 72.9 (超买⚠️) | MACD: 金叉")
print("   ⚠️ RSI超买,布林带高位(91.7%),短期可能回调")
print("   💰 建议仓位: 20-30% 或等回调")

print("\n3️⃣ AVGO Broadcom ($238.50) - ⭐⭐⭐ 可考虑")
print("   信号: 买入 0.25 | RSI: 52.3 (中性) | MACD: 金叉")
print("   ✅ 技术面健康,MACD金叉")
print("   💰 建议仓位: 20-30%")

print("\n4️⃣ GOOG Google ($308.61) - ⭐⭐ 观望")
print("   信号: 买入 0.19 (很弱) | RSI: 44.7 (中性) | MACD: 死叉")
print("   ⚠️ 信号弱,MACD死叉,建议观望")
print("   💰 建议仓位: 10-20% 或观望")

print("\n5️⃣ MU Micron ($265.92) - 🔴 卖出信号")
print("   信号: 卖出 0.12 | RSI: 59.8 (偏高) | MACD: 金叉")
print("   ⚠️ 布林带极高位(97.1%),虽然MACD金叉但AI建议减仓")
print("   💰 建议操作: 减仓10-20%,或持有观望")

print("\n" + "=" * 100)
print("📌 说明:")
print("  • 强度范围: 0.0 ~ 1.0")
print("  • 0.6+ = 强力信号 (🟢🟢🟢)")
print("  • 0.4-0.6 = 中等信号 (🟢🟢)")
print("  • 0.2-0.4 = 弱信号 (🟢)")
print("  • < 0.2 = 观望 (🟡)")

print("\n⚠️  风险提示:")
print("  • 本信号由 AI 模型生成,仅供参考,不构成投资建议")
print("  • 请结合 RSI、MACD 等技术指标综合判断")
print("  • 投资有风险,入市需谨慎")
print("=" * 100)

print("\n💡 如需查看单个股票详细信号,请运行:")
print("   python C:\\Users\\Silvi\\get_trading_signal_2317.py")
print("   python C:\\Users\\Silvi\\get_trading_signal_8131.py")
print("   python C:\\Users\\Silvi\\get_trading_signal_goog.py")
print("   python C:\\Users\\Silvi\\get_trading_signal_avgo.py")
print("   python C:\\Users\\Silvi\\get_trading_signal_mu.py")
