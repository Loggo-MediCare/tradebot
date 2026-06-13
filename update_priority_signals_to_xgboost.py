"""
批次更新優先美股信號文件從 PPO 到 XGBoost
股票: MSFT, GOOGL, AMD, INTC, META, ORCL
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os

priority_stocks = [
    {'ticker': 'MSFT', 'name': 'Microsoft'},
    {'ticker': 'GOOGL', 'name': 'Google'},
    {'ticker': 'AMD', 'name': 'AMD'},
    {'ticker': 'INTC', 'name': 'Intel'},
    {'ticker': 'META', 'name': 'Meta'},
    {'ticker': 'ORCL', 'name': 'Oracle'},
]

# AMZN 模板文件
template_file = 'get_trading_signal_amzn.py'

print("=" * 80)
print("批次更新優先美股信號文件到 XGBoost")
print("=" * 80)

for stock in priority_stocks:
    ticker = stock['ticker']
    name = stock['name']

    signal_file = f'get_trading_signal_{ticker.lower()}.py'

    print(f"\n處理: {ticker} ({name})")

    # 讀取模板
    with open(template_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 替換股票代碼和名稱
    content = content.replace('AMZN', ticker)
    content = content.replace('Amazon', name)
    content = content.replace('amzn', ticker.lower())

    # 針對不同股票調整專屬策略部分
    # 移除 AMZN 專屬的1303相關性分析
    if ticker != 'AMZN':
        # 找到專屬策略區塊並移除
        start_marker = "# ==========================================\n        # 🎯 AMZN 专属策略增强 (基于相关性与机构资金)\n        # =========================================="
        end_marker = "        # ==========================================\n\n\n"

        start_idx = content.find(start_marker)
        if start_idx != -1:
            end_idx = content.find(end_marker, start_idx)
            if end_idx != -1:
                # 替換為通用的機構資金分析
                generic_strategy = '''# ==========================================
        # 🎯 ''' + name + ''' 机构资金分析
        # ==========================================
        print("\\n" + "=" * 80)
        print("🎯 ''' + ticker + ''' ''' + name + ''' 机构资金分析")
        print("=" * 80)

        # OBV机构资金确认 (权值股特有)
        obv_current = df['obv'].iloc[-1] if 'obv' in df.columns else 0
        obv_ma20 = df['obv_ma20'].iloc[-1] if 'obv_ma20' in df.columns else 0
        obv_5d_avg = df['obv'].tail(5).mean() if 'obv' in df.columns and len(df) >= 5 else 0

        institutional_inflow = False
        if obv_current > obv_ma20 and obv_5d_avg > obv_ma20:
            institutional_inflow = True
            buy_score += 20
            buy_reasons.append("💰 OBV持续向上，机构资金进驻确认")

        print(f"OBV 机构资金分析 (权值股关键指标):")
        print(f"  当前OBV:     {obv_current:,.0f}")
        print(f"  OBV MA20:    {obv_ma20:,.0f}")
        print(f"  5日均值:     {obv_5d_avg:,.0f}")
        print(f"  机构进驻:    {'✅ 是 (OBV持续突破MA20)' if institutional_inflow else '❌ 否 (等待OBV确认)'}")

        # 目标价位区间（如果有分析师数据）
        if target_price is not None:
            target_zone_low = target_price
            target_zone_high = target_high if target_high else target_price * 1.1
            distance_to_target = ((target_zone_low - current_price) / current_price) * 100

            print(f"\\n目标价位分析:")
            print(f"  当前价格:     ${current_price:.2f}")
            print(f"  分析师目标:   ${target_zone_low:.2f} - ${target_zone_high:.2f}")
            print(f"  潜在空间:     {distance_to_target:+.1f}% {'(已达标)' if distance_to_target <= 0 else '(上涨空间)'}")

            # 如果已接近或超过目标区，降低买入评分
            if current_price >= target_zone_high:
                buy_score -= 15
                buy_warnings.append(f"⚠️ 价格已达目标区上限 ${target_zone_high:.2f}, 建议获利了结")
            elif current_price >= target_zone_low:
                buy_score -= 5
                buy_warnings.append(f"价格已进入目标区 ${target_zone_low:.2f}-{target_zone_high:.2f}, 注意压力")

        print("\\n" + "=" * 80)
        # ==========================================


'''
                content = content[:start_idx] + generic_strategy + content[end_idx + len(end_marker):]

    # 寫入新文件
    with open(signal_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[OK] Updated: {signal_file}")

print("\n" + "=" * 80)
print("全部完成!")
print("=" * 80)
print(f"成功更新 {len(priority_stocks)} 個信號文件")
