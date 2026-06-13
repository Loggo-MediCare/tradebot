"""
Upgrade money_flow_strength in all get_trading_signal_*.py files
- Add up_volume / down_volume distinction
- Add money_flow = volume * (close - open)
- Add capital_inflow detection
"""
import glob
import re

OLD_FUNCTION = '''def money_flow_strength(df):
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
    return strong_money, volume_ratio'''

NEW_FUNCTION = '''def money_flow_strength(df):
    """分析资金流向强度 (进阶版: up/down volume + money flow)"""
    if len(df) < 20:
        return False, 1.0, {}

    # 量比
    volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]

    # Up Volume vs Down Volume (基于 close vs open)
    recent = df.tail(30).copy()
    recent['up_volume'] = recent.apply(lambda x: x['volume'] if x['close'] > x['open'] else 0, axis=1)
    recent['down_volume'] = recent.apply(lambda x: x['volume'] if x['close'] <= x['open'] else 0, axis=1)
    up_vol = recent['up_volume'].sum()
    down_vol = recent['down_volume'].sum()
    up_down_ratio = up_vol / (down_vol + 1e-10)

    # Money Flow = volume * (close - open)
    recent['money_flow'] = recent['volume'] * (recent['close'] - recent['open'])
    net_money_flow_30d = recent['money_flow'].sum()
    net_money_flow_5d = recent['money_flow'].tail(5).sum()

    # Capital inflow: 放量 + 收阳
    latest = df.iloc[-1]
    capital_inflow = (volume_ratio > 1.5) and (latest['close'] > latest['open'])

    # OBV
    obv_now = df['obv'].iloc[-1] if 'obv' in df.columns else 0
    obv_ma = df['obv_ma20'].iloc[-1] if 'obv_ma20' in df.columns else 0
    obv_bullish = obv_now > obv_ma

    # 综合判断
    strong_money = (
        (capital_inflow or (up_down_ratio > 1.3 and volume_ratio > 1.0)) and
        obv_bullish
    )

    details = {
        'up_volume_30d': int(up_vol),
        'down_volume_30d': int(down_vol),
        'up_down_ratio': round(up_down_ratio, 2),
        'net_money_flow_30d': net_money_flow_30d,
        'net_money_flow_5d': net_money_flow_5d,
        'capital_inflow': capital_inflow,
        'obv_bullish': obv_bullish,
    }

    return strong_money, volume_ratio, details'''

# Update explosive_trend_filter to handle new 3-return value
OLD_EXPLOSIVE = '''    strong_money, vol_ratio = money_flow_strength(df)'''
NEW_EXPLOSIVE = '''    strong_money, vol_ratio, flow_details = money_flow_strength(df)'''

# Update the return dict of explosive_trend_filter to include flow_details
OLD_RETURN = '''    return {
        "explosive": explosive,
        "volume_ratio": vol_ratio,
        "cycle_phase": cycle_phase,
        "money_inflow": strong_money,
        "trend_accelerating": accelerating'''

NEW_RETURN = '''    return {
        "explosive": explosive,
        "volume_ratio": vol_ratio,
        "cycle_phase": cycle_phase,
        "money_inflow": strong_money,
        "trend_accelerating": accelerating,
        "flow_details": flow_details'''

# Old print section
OLD_PRINT = '''    print(f"资金流入状态: {\'✅ 强势\' if explosion[\'money_inflow\'] else \'❌ 弱势\'}")
    print(f"趋势加速状态: {\'✅ 加速中\' if explosion[\'trend_accelerating\'] else \'❌ 减速中\'}")
    print(f"周期阶段: {explosion[\'cycle_phase\']}")
    print(f"量比: {explosion[\'volume_ratio\']:.2f}x")'''

NEW_PRINT = '''    flow = explosion.get('flow_details', {})
    print(f"资金流入状态: {'✅ 强势' if explosion['money_inflow'] else '❌ 弱势'}")
    if flow:
        print(f"   Up Volume (30d):   {flow.get('up_volume_30d', 0):,}")
        print(f"   Down Volume (30d): {flow.get('down_volume_30d', 0):,}")
        print(f"   Up/Down Ratio:     {flow.get('up_down_ratio', 0):.2f}x  {'[多方主导]' if flow.get('up_down_ratio', 0) > 1.3 else '[空方主导]' if flow.get('up_down_ratio', 0) < 0.7 else '[均衡]'}")
        mf_30 = flow.get('net_money_flow_30d', 0)
        mf_5 = flow.get('net_money_flow_5d', 0)
        print(f"   Net Money Flow(30d): {'+'if mf_30>0 else ''}{mf_30:,.0f}  {'[净流入]' if mf_30 > 0 else '[净流出]'}")
        print(f"   Net Money Flow(5d):  {'+'if mf_5>0 else ''}{mf_5:,.0f}  {'[近期流入]' if mf_5 > 0 else '[近期流出]'}")
        print(f"   即时资金流入:  {'✅ 放量收阳' if flow.get('capital_inflow') else '❌ 未确认'}")
        print(f"   OBV趋势:      {'✅ 多头' if flow.get('obv_bullish') else '❌ 空头'}")
    print(f"趋势加速状态: {'✅ 加速中' if explosion['trend_accelerating'] else '❌ 减速中'}")
    print(f"周期阶段: {explosion['cycle_phase']}")
    print(f"量比: {explosion['volume_ratio']:.2f}x")'''

files = sorted(glob.glob('get_trading_signal_*.py'))
updated = 0
errors = []

for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            content = fh.read()

        original = content

        # 1. Replace money_flow_strength function
        if OLD_FUNCTION in content:
            content = content.replace(OLD_FUNCTION, NEW_FUNCTION)

        # 2. Replace explosive_trend_filter call
        if OLD_EXPLOSIVE in content:
            content = content.replace(OLD_EXPLOSIVE, NEW_EXPLOSIVE)

        # 3. Replace return dict
        if OLD_RETURN in content:
            content = content.replace(OLD_RETURN, NEW_RETURN)

        # 4. Replace print section (use regex for flexibility with quotes)
        old_print_pattern = (
            '    print(f"资金流入状态: {\'✅ 强势\' if explosion[\'money_inflow\'] else \'❌ 弱势\'}")\n'
            '    print(f"趋势加速状态: {\'✅ 加速中\' if explosion[\'trend_accelerating\'] else \'❌ 减速中\'}")\n'
            '    print(f"周期阶段: {explosion[\'cycle_phase\']}")\n'
            '    print(f"量比: {explosion[\'volume_ratio\']:.2f}x")'
        )

        new_print_block = """    flow = explosion.get('flow_details', {})
    print(f"资金流入状态: {'✅ 强势' if explosion['money_inflow'] else '❌ 弱势'}")
    if flow:
        print(f"   Up Volume (30d):   {flow.get('up_volume_30d', 0):,}")
        print(f"   Down Volume (30d): {flow.get('down_volume_30d', 0):,}")
        print(f"   Up/Down Ratio:     {flow.get('up_down_ratio', 0):.2f}x  {'[多方主导]' if flow.get('up_down_ratio', 0) > 1.3 else '[空方主导]' if flow.get('up_down_ratio', 0) < 0.7 else '[均衡]'}")
        mf_30 = flow.get('net_money_flow_30d', 0)
        mf_5 = flow.get('net_money_flow_5d', 0)
        print(f"   Net Money Flow(30d): {'+'if mf_30>0 else ''}{mf_30:,.0f}  {'[净流入]' if mf_30 > 0 else '[净流出]'}")
        print(f"   Net Money Flow(5d):  {'+'if mf_5>0 else ''}{mf_5:,.0f}  {'[近期流入]' if mf_5 > 0 else '[近期流出]'}")
        print(f"   即时资金流入:  {'✅ 放量收阳' if flow.get('capital_inflow') else '❌ 未确认'}")
        print(f"   OBV趋势:      {'✅ 多头' if flow.get('obv_bullish') else '❌ 空头'}")
    print(f"趋势加速状态: {'✅ 加速中' if explosion['trend_accelerating'] else '❌ 减速中'}")
    print(f"周期阶段: {explosion['cycle_phase']}")
    print(f"量比: {explosion['volume_ratio']:.2f}x")"""

        if old_print_pattern in content:
            content = content.replace(old_print_pattern, new_print_block)

        if content != original:
            with open(f, 'w', encoding='utf-8') as fh:
                fh.write(content)
            updated += 1
        else:
            errors.append(f"{f}: no changes made")
    except Exception as e:
        errors.append(f"{f}: {e}")

print(f"Updated: {updated}/{len(files)} files")
if errors:
    print(f"\nSkipped/errors ({len(errors)}):")
    for e in errors[:10]:
        print(f"  {e}")
