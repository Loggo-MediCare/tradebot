"""
Integrate Explosion Detection (Breakout Market Detection) into all get_trading*.py files
"""
import os
import re
import sys
import io
from pathlib import Path

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# The explosion detection functions to add
EXPLOSION_FUNCTIONS = '''
# ==========================================
# 新增：资金流向分析函数
# ==========================================
def calculate_obv(df):
    """计算OBV (能量潮指标)"""
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    df['obv_ma20'] = pd.Series(obv).rolling(20).mean()
    return df

def money_flow_strength(df):
    """分析资金流向强度"""
    if len(df) < 20:
        return False, 1.0

    obv_now = df['obv'].iloc[-1]
    obv_ma = df['obv_ma20'].iloc[-1]
    volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]

    strong_money = (
        obv_now > obv_ma and
        volume_ratio > 1.3
    )
    return strong_money, volume_ratio

def detect_memory_cycle_phase(df):
    """检测内存周期阶段（适用于芯片股）"""
    if len(df) < 200:
        return "NEUTRAL"

    ma50 = df['sma_50']
    ma200 = df['sma_200']
    price = df['close'].iloc[-1]

    # 週期初升段
    early_upcycle = (
        price > ma50.iloc[-1] and
        ma50.iloc[-1] > ma200.iloc[-1] and
        ma50.diff().iloc[-1] > 0
    )

    # 高檔末升段
    late_cycle = (
        price > ma50.iloc[-1] * 1.25 and
        ma50.diff().iloc[-1] < ma50.diff().iloc[-5] if len(df) >= 5 else False
    )

    if early_upcycle:
        return "EARLY_UPCYCLE"   # 🔥最會噴的階段
    elif late_cycle:
        return "LATE_CYCLE"
    else:
        return "NEUTRAL"

def trend_acceleration(df):
    """检测趋势加速"""
    if len(df) < 30:
        return False

    sma10 = df['sma_10']
    sma30 = df['sma_30']
    slope10 = sma10.diff().iloc[-1]
    slope30 = sma30.diff().iloc[-1]
    price = df['close'].iloc[-1]

    accelerating = (
        slope10 > slope30 and
        slope10 > 0 and
        price > sma10.iloc[-1]
    )
    return accelerating

def explosive_trend_filter(df):
    """爆发行情过滤器"""
    strong_money, vol_ratio = money_flow_strength(df)
    cycle_phase = detect_memory_cycle_phase(df)
    accelerating = trend_acceleration(df)

    explosive = (
        strong_money and
        accelerating and
        cycle_phase == "EARLY_UPCYCLE"
    )

    return {
        "explosive": explosive,
        "volume_ratio": vol_ratio,
        "cycle_phase": cycle_phase,
        "money_inflow": strong_money,
        "trend_accelerating": accelerating
    }
'''

# The explosion detection usage code
EXPLOSION_DETECTION_CODE = '''
    # 8. 爆发行情检测（主升段分析）
    print("\\n" + "=" * 80)
    print("🚀 爆发行情检测 (主升段分析)")
    print("=" * 80)

    explosion = explosive_trend_filter(df)
    print(f"资金流入状态: {'✅ 强势' if explosion['money_inflow'] else '❌ 弱势'}")
    print(f"趋势加速状态: {'✅ 加速中' if explosion['trend_accelerating'] else '❌ 减速中'}")
    print(f"周期阶段: {explosion['cycle_phase']}")
    print(f"量比: {explosion['volume_ratio']:.2f}x")

    if explosion["explosive"]:
        print("\\n🚀 主升段爆发行情侦测!")
        print("📌 爆发行情特征:")
        print("   • 资金强势流入 (OBV > 20日均线)")
        print("   • 趋势加速 (10日均线斜率 > 30日均线斜率)")
        print("   • 处于周期初升段 (EARLY_UPCYCLE)")
        print("   • 量能放大 (量比 > 1.3x)")
'''

def check_if_has_explosion_functions(content):
    """Check if file already has explosion detection functions"""
    return 'explosive_trend_filter' in content and 'money_flow_strength' in content

def add_obv_to_technical_indicators(content):
    """Add OBV calculation to add_technical_indicators function"""
    # Find the add_technical_indicators function
    pattern = r'(def add_technical_indicators\(df\):.*?)(    df = df\.bfill\(\)\.ffill\(\)\s+return df)'

    match = re.search(pattern, content, re.DOTALL)
    if match:
        before_return = match.group(1)
        return_statement = match.group(2)

        # Check if OBV is already added
        if 'calculate_obv' not in before_return:
            # Add OBV calculation before the return statement
            new_content = before_return + '''
    # 计算OBV (能量潮指标)
    df = calculate_obv(df)

    ''' + return_statement
            content = content.replace(match.group(0), new_content)

    return content

def add_explosion_to_get_trading_signal(content, ticker):
    """Add explosion detection to get_trading_signal function"""
    # Find where to insert the explosion detection (before signal generation)
    # Look for "生成交易建议" or "AI 交易信号"
    pattern = r'(\n    # 9\. 生成交易建议\s+print\("\\n" \+ "=" \* 80\))'

    match = re.search(pattern, content)
    if match:
        # Insert explosion detection before signal generation
        insertion_point = match.start()
        new_content = content[:insertion_point] + '\n' + EXPLOSION_DETECTION_CODE + content[insertion_point:]
        return new_content

    return content

def update_add_technical_indicators_sma200(content):
    """Ensure SMA 200 is added to technical indicators"""
    # Find the add_technical_indicators function
    pattern = r"(    df\['sma_50'\] = df\['close'\]\.rolling\(50\)\.mean\(\))"

    if pattern in content and "df['sma_200']" not in content:
        replacement = r"\1\n    df['sma_200'] = df['close'].rolling(200).mean()  # 添加200日均线"
        content = re.sub(pattern, replacement, content)

    return content

def integrate_explosion_detection(file_path):
    """Integrate explosion detection into a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Skip if already has explosion functions
        if check_if_has_explosion_functions(content):
            return True, "Already has explosion detection"

        original_content = content

        # Step 1: Add SMA 200 to technical indicators if not present
        content = update_add_technical_indicators_sma200(content)

        # Step 2: Add explosion detection functions after add_technical_indicators
        # Find the end of add_technical_indicators function
        pattern = r'(def add_technical_indicators\(df\):.*?return df\n)'
        match = re.search(pattern, content, re.DOTALL)

        if match:
            insertion_point = match.end()
            content = content[:insertion_point] + '\n' + EXPLOSION_FUNCTIONS + '\n' + content[insertion_point:]

        # Step 3: Add OBV to add_technical_indicators
        content = add_obv_to_technical_indicators(content)

        # Step 4: Add explosion detection to get_trading_signal function
        # Extract ticker from filename
        ticker_match = re.search(r'get_trading_signal_(.+)\.py', file_path.name)
        ticker = ticker_match.group(1) if ticker_match else 'unknown'
        content = add_explosion_to_get_trading_signal(content, ticker)

        # Step 5: Add explosion detection in buy signal scoring
        # Find the buy signal section and add explosion bonus
        buy_pattern = r'(        # 加入MA50斜率評分調整\s+ma50_slope_adjustment = get_ma50_slope_score_adjustment\(ma50_slope_info\)\s+buy_score \+= ma50_slope_adjustment)'

        if re.search(buy_pattern, content):
            explosion_bonus = '''

        # 加入爆发行情评分调整
        if explosion["explosive"]:
            buy_score += 25  # 爆发行情额外加分
            buy_reasons.append(f"🚀 爆发行情确认: 主升段初期")
            buy_reasons.append(f"资金强势流入 (OBV > MA20)")
            buy_reasons.append(f"趋势加速 (10日斜率 > 30日斜率)")'''

            content = re.sub(buy_pattern, r'\1' + explosion_bonus, content)

        # Step 6: Add explosion override for sell signals
        # Find the sell signal section
        sell_pattern = r'(    elif action_value < -0\.1:)'

        if re.search(sell_pattern, content):
            explosion_override = '''
        # 先检查是否为爆发行情，如果是则覆盖卖出信号
        if explosion["explosive"]:
            signal = "强势持有 (HOLD - TREND EXPLOSION)"
            signal_emoji = "🚀"
            strength = abs(action_value)
            suggested_price_low = current_price
            suggested_price_high = current_price

            print("\\n🚀 主升段爆发行情侦测!")
            print(f"资金流入: {explosion['money_inflow']}")
            print(f"趋势加速: {explosion['trend_accelerating']}")
            print(f"周期位置: {explosion['cycle_phase']}")
            print(f"量比: {explosion['volume_ratio']:.2f}x")

            print("\\n📌 操作策略:")
            print("   • 不卖出 (主升段爆发行情)")
            print("   • 回调不破均线继续抱")
            print("   • 使用追踪止损代替固定止损")
            print("   • 关注 OBV 资金流向指标")
            print("   • 设置移动止盈: 跌破 10 日均线减半仓")

            # 跳过卖出评分逻辑
            skip_sell_scoring = True
        else:
            skip_sell_scoring = False

        if not skip_sell_scoring:'''

            # Replace the sell signal line with explosion override
            content = re.sub(sell_pattern, r'\1\n' + explosion_override, content)

        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Successfully integrated"
        else:
            return False, "No changes needed"

    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main function to integrate explosion detection into all get_trading*.py files"""
    print("=" * 80)
    print("🚀 Integrating 爆发行情检测 into all get_trading*.py files")
    print("=" * 80)

    # Find all get_trading*.py files
    current_dir = Path('.')
    files = list(current_dir.glob('get_trading_signal_*.py'))

    if not files:
        print("❌ No get_trading_signal_*.py files found")
        return

    print(f"Found {len(files)} files to process\n")

    success_count = 0
    skip_count = 0
    error_count = 0

    for file_path in sorted(files):
        print(f"Processing {file_path.name}...", end=" ")
        success, message = integrate_explosion_detection(file_path)

        if success:
            if "Already has" in message:
                print(f"[SKIP] {message}")
                skip_count += 1
            else:
                print(f"[OK] {message}")
                success_count += 1
        else:
            print(f"[ERROR] {message}")
            error_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("Integration Summary")
    print("=" * 80)
    print(f"Total files:        {len(files)}")
    print(f"Successfully added: {success_count}")
    print(f"Already had:        {skip_count}")
    print(f"Errors:             {error_count}")
    print("=" * 80)

    if success_count > 0:
        print(f"\n✅ Successfully integrated 爆发行情检测 into {success_count} files!")
    else:
        print("\n⚠️  All files already have explosion detection or no changes needed")

if __name__ == "__main__":
    main()
