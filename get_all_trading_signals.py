"""
所有股票 AI 交易信号汇总
======================================
一次性查看所有训练模型的交易信号
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from datetime import datetime
import subprocess
import json

def load_signal_module(module_path, module_name):
    """动态加载信号生成模块"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"❌ 加载模块失败 {module_name}: {e}")
        return None

def get_signal_emoji(strength):
    """根据信号强度返回表情"""
    if strength >= 0.6:
        return "🟢🟢🟢"  # 强力买入/卖出
    elif strength >= 0.4:
        return "🟢🟢"    # 买入/卖出
    elif strength >= 0.2:
        return "🟢"      # 弱买入/卖出
    else:
        return "🟡"      # 持有

def format_signal_table(signals):
    """格式化信号表格"""
    print("\n" + "=" * 100)
    print("📊 所有股票 AI 交易信号汇总")
    print("=" * 100)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    # 表头
    print(f"\n{'股票':<12} {'信号':<12} {'强度':<8} {'当前价格':<15} {'RSI':<8} {'MACD':<10} {'评级':<10}")
    print("-" * 100)

    # 按信号强度排序
    sorted_signals = sorted(signals, key=lambda x: x.get('strength', 0), reverse=True)

    for sig in sorted_signals:
        symbol = sig['symbol']
        signal = sig['signal']
        strength = sig.get('strength', 0)
        price = sig['current_price']
        rsi = sig.get('rsi', 0)
        macd = sig.get('macd', 0)
        emoji = get_signal_emoji(strength)

        # 货币符号
        currency = 'NT$' if '.TW' in symbol else '$'

        # 信号颜色标记
        if '买入' in signal:
            signal_display = f"🟢 {signal}"
        elif '卖出' in signal:
            signal_display = f"🔴 {signal}"
        else:
            signal_display = f"🟡 {signal}"

        print(f"{symbol:<12} {signal_display:<20} {strength:<8.2f} {currency}{price:<14.2f} {rsi:<8.1f} {macd:<10.2f} {emoji}")

    print("=" * 100)

    # 投资建议
    print("\n💡 投资建议排名:")
    print("-" * 100)

    for i, sig in enumerate(sorted_signals[:5], 1):  # 只显示前5名
        symbol = sig['symbol']
        strength = sig.get('strength', 0)
        signal = sig['signal']
        price = sig['current_price']
        currency = 'NT$' if '.TW' in symbol else '$'

        if strength >= 0.4:
            recommendation = "✅ 优先考虑"
        elif strength >= 0.2:
            recommendation = "⚠️ 谨慎考虑"
        else:
            recommendation = "🟡 观望为主"

        print(f"{i}. {symbol:<10} - {signal:<15} 强度: {strength:.2f}  价格: {currency}{price:.2f}  {recommendation}")

    print("=" * 100)

def main():
    """主程序"""
    print("🤖 正在加载所有股票的 AI 交易信号...")
    print("=" * 100)

    signals = []

    # 定义所有股票及其模块路径
    stocks = [
        ('2317.TW', r'C:\Users\Silvi\get_trading_signal_2317.py', '2317'),
        ('8131.TW', r'C:\Users\Silvi\get_trading_signal_8131.py', '8131'),
        ('GOOG', r'C:\Users\Silvi\get_trading_signal_goog.py', 'goog'),
        ('AVGO', r'C:\Users\Silvi\get_trading_signal_avgo.py', 'avgo'),
        ('MU', r'C:\Users\Silvi\get_trading_signal_mu.py', 'mu'),
    ]

    for symbol, module_path, module_name in stocks:
        print(f"\n📊 获取 {symbol} 信号...")

        # 检查模块文件是否存在
        if not os.path.exists(module_path):
            print(f"   ⚠️ 信号生成器不存在: {module_path}")
            continue

        try:
            # 加载模块
            module = load_signal_module(module_path, module_name)
            if module is None:
                continue

            # 获取信号
            result = module.get_trading_signal()

            if result:
                signals.append(result)
                print(f"   ✅ {symbol} 信号获取成功: {result['signal']} (强度: {result.get('strength', 0):.2f})")
            else:
                print(f"   ❌ {symbol} 信号获取失败")

        except Exception as e:
            print(f"   ❌ {symbol} 处理失败: {e}")

    # 显示汇总表格
    if signals:
        format_signal_table(signals)

        print("\n📌 说明:")
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
    else:
        print("\n❌ 没有获取到任何信号")

if __name__ == "__main__":
    main()
