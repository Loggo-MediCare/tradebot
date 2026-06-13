"""
批量运行所有台股的交易信号生成器
================================
自动运行所有已训练的台股模型的交易信号
"""

import subprocess
import sys
import io
import os
import re
from datetime import datetime

# 修复 Windows 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 强制输出刷新（避免缓冲导致终端只显示最后几个股票）
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUNBUFFERED'] = '1'

# 台股信号生成器 (updated 2025-01)
SIGNAL_SCRIPTS = [
    {'file': 'get_trading_signal_2317.py', 'name': '2317 鴻海'},
    {'file': 'get_trading_signal_2330.py', 'name': '2330 台積電'},
    {'file': 'get_trading_signal_2382.py', 'name': '2382 廣達'},
    {'file': 'get_trading_signal_3036.py', 'name': '3036 文曄'},
    {'file': 'get_trading_signal_2357.py', 'name': '2357 華碩'},
    {'file': 'get_trading_signal_3711.py', 'name': '3711 日月光投控'},
    {'file': 'get_trading_signal_2308.py', 'name': '2308 台達電'},
    {'file': 'get_trading_signal_2454.py', 'name': '2454 聯發科'},
    
    {'file': 'get_trading_signal_1101.py', 'name': '1101 台泥'},
    {'file': 'get_trading_signal_7805.py', 'name': '7805 威廉通訊'},
    {'file': 'get_trading_signal_6187.py', 'name': '6187 萬潤'},
    {'file': 'get_trading_signal_1301.py', 'name': '1301 台塑'},
    {'file': 'get_trading_signal_1303.py', 'name': '1303 南亞'},
    {'file': 'get_trading_signal_1504.py', 'name': '1504 東元'},
    {'file': 'get_trading_signal_1514.py', 'name': '1514 亞力'},
    {'file': 'get_trading_signal_1513.py', 'name': '1513 中興電'},
    {'file': 'get_trading_signal_1519.py', 'name': '1519 華城'},
    {'file': 'get_trading_signal_1560.py', 'name': '1560 中砂'},
    {'file': 'get_trading_signal_1605.py', 'name': '1605 華新'},
    {'file': 'get_trading_signal_1802.py', 'name': '1802 台玻'},
    {'file': 'get_trading_signal_2303.py', 'name': '2303 聯電'},
   
    {'file': 'get_trading_signal_2301.py', 'name': '2301 光寶科技'},
    {'file': 'get_trading_signal_2313.py', 'name': '2313 華通'},
    
    {'file': 'get_trading_signal_2324.py', 'name': '2324 仁寶'},
    {'file': 'get_trading_signal_2327.py', 'name': '2327 國巨'},
    {'file': 'get_trading_signal_2377.py', 'name': '2377 微星'},
  
    {'file': 'get_trading_signal_2331.py', 'name': '2331 精英'},
    {'file': 'get_trading_signal_2337.py', 'name': '2337 旺宏'},
    {'file': 'get_trading_signal_2344.py', 'name': '2344 華邦電'},
    {'file': 'get_trading_signal_2345.py', 'name': '2345 智邦'},
    {'file': 'get_trading_signal_2360.py', 'name': '2360 致茂'},
    {'file': 'get_trading_signal_2363.py', 'name': '2363 矽統'},
    {'file': 'get_trading_signal_2367.py', 'name': '2367 燿華'},
    {'file': 'get_trading_signal_2368.py', 'name': '2368 金像電'},
    {'file': 'get_trading_signal_2376.py', 'name': '2376 技嘉'},
    
    {'file': 'get_trading_signal_2383.py', 'name': '2383 台光電'},
    {'file': 'get_trading_signal_2385.py', 'name': '2385 群光'},
    {'file': 'get_trading_signal_2408.py', 'name': '2408 南亞科'},
    {'file': 'get_trading_signal_2409.py', 'name': '2409 友達'},
    {'file': 'get_trading_signal_2449.py', 'name': '2449 京元電子'},
    {'file': 'get_trading_signal_2451.py', 'name': '2451 創見資訊'},
    
    {'file': 'get_trading_signal_2489.py', 'name': '2489 瑞軒'},
    {'file': 'get_trading_signal_2603.py', 'name': '2603 長榮'},
    {'file': 'get_trading_signal_2634.py', 'name': '2634 漢翔'},
    {'file': 'get_trading_signal_2881.py', 'name': '2881 富邦金'},
    {'file': 'get_trading_signal_2884.py', 'name': '2884 玉山金'},
    {'file': 'get_trading_signal_3004.py', 'name': '3004 豐達科'},
    {'file': 'get_trading_signal_3006.py', 'name': '3006 晶豪科'},
    {'file': 'get_trading_signal_3017.py', 'name': '3017 奇鋐'},
    {'file': 'get_trading_signal_3022.py', 'name': '3022 威強電'},
    {'file': 'get_trading_signal_3030.py', 'name': '3030 德律'},
    {'file': 'get_trading_signal_3037.py', 'name': '3037 欣興'},
    {'file': 'get_trading_signal_3081.py', 'name': '3081 聯亞 (矽光子)'},
    {'file': 'get_trading_signal_3135.py', 'name': '3135 晶技'},
    {'file': 'get_trading_signal_3149.py', 'name': '3149 正達'},
    {'file': 'get_trading_signal_3231.py', 'name': '3231 緯創'},
    {'file': 'get_trading_signal_3363.py', 'name': '3363 上詮'},
    {'file': 'get_trading_signal_3443.py', 'name': '3443 創意'},
    {'file': 'get_trading_signal_3449.py', 'name': '3449 鈺德'},
    {'file': 'get_trading_signal_3481.py', 'name': '3481 群創'},
    {'file': 'get_trading_signal_3491.py', 'name': '3491 昇達科'},
    {'file': 'get_trading_signal_3532.py', 'name': '3532 台勝科'},
    {'file': 'get_trading_signal_3653.py', 'name': '3653 健策'},
    {'file': 'get_trading_signal_3661.py', 'name': '3661 世芯-KY'},
  
    {'file': 'get_trading_signal_3715.py', 'name': '3715 定穎投控'},
    {'file': 'get_trading_signal_4540.py', 'name': '4540 全球傳動'},
    {'file': 'get_trading_signal_4722.py', 'name': '4722 国精化'},
    {'file': 'get_trading_signal_4746.py', 'name': '4746 台耀'},
    {'file': 'get_trading_signal_4916.py', 'name': '4916 事欣科'},
    {'file': 'get_trading_signal_4938.py', 'name': '4938 和碩'},
    {'file': 'get_trading_signal_6209.py', 'name': '6209 今國光'},
    {'file': 'get_trading_signal_6239.py', 'name': '6239 力成'},
    {'file': 'get_trading_signal_6269.py', 'name': '6269 台郡'},
    {'file': 'get_trading_signal_6282.py', 'name': '6282 康舒'},
    {'file': 'get_trading_signal_6285.py', 'name': '6285 啟碁'},
    {'file': 'get_trading_signal_6442.py', 'name': '6442 光聖'},
    {'file': 'get_trading_signal_6443.py', 'name': '6443 元晶'},
    {'file': 'get_trading_signal_6446.py', 'name': '6446 藥華藥'},
    {'file': 'get_trading_signal_6477.py', 'name': '6477 安集'},
    {'file': 'get_trading_signal_6515.py', 'name': '6515 穎崴'},
    {'file': 'get_trading_signal_6668.py', 'name': '6668 中揚光'},
    {'file': 'get_trading_signal_6669.py', 'name': '6669 緯穎'},
    {'file': 'get_trading_signal_6770.py', 'name': '6770 力積電'},
    {'file': 'get_trading_signal_6781.py', 'name': '6781 AES-KY'},
    {'file': 'get_trading_signal_6805.py', 'name': '6805 富世達'},
    {'file': 'get_trading_signal_7769.py', 'name': '7769 聚和'},
    {'file': 'get_trading_signal_8021.py', 'name': '8021 尖點'},
    {'file': 'get_trading_signal_8033.py', 'name': '8033 雷虎'},
    {'file': 'get_trading_signal_8046.py', 'name': '8046 南電'},
    {'file': 'get_trading_signal_8112.py', 'name': '8112 至上'},
    {'file': 'get_trading_signal_8110.py', 'name': '8110 華東'},
    {'file': 'get_trading_signal_8131.py', 'name': '8131 福懋科'},
    {'file': 'get_trading_signal_8150.py', 'name': '8150 南茂'},
    {'file': 'get_trading_signal_8210.py', 'name': '8210 勤誠'},
    {'file': 'get_trading_signal_8222.py', 'name': '8222 寶一'},
    {'file': 'get_trading_signal_8499.py', 'name': '8499 鼎炫-KY'},
    {'file': 'get_trading_signal_8996.py', 'name': '8996 高力'},
    
    {'file': 'get_trading_signal_3046.py', 'name': '3046 建碁'},
    {'file': 'get_trading_signal_6643.py', 'name': '6643 M31'},
    {'file': 'get_trading_signal_3152.py', 'name': '3152 璟德'},
    {'file': 'get_trading_signal_2455.py', 'name': '2455 全新'},
    {'file': 'get_trading_signal_6706.py', 'name': '6706 惠特'},
   
    {'file': 'get_trading_signal_8043.py', 'name': '8043 蜜望實'},
    {'file': 'get_trading_signal_7610.py', 'name': '7610 聯友金屬'},
    {'file': 'get_trading_signal_6834.py', 'name': '6834 天二科技'},
    {'file': 'get_trading_signal_6742.py', 'name': '6742 澤米'},
    {'file': 'get_trading_signal_5285.py', 'name': '5285 界霖'},
    {'file': 'get_trading_signal_3105.py', 'name': '3105 穩懋'},
    {'file': 'get_trading_signal_3694.py', 'name': '3694 海華'},
    {'file': 'get_trading_signal_8044.py', 'name': '8044 網家'},
    {'file': 'get_trading_signal_3563.py', 'name': '3563 牧德'},
    {'file': 'get_trading_signal_3504.py', 'name': '3504 揚明光'},
    {'file': 'get_trading_signal_5328.py', 'name': '5328 華容'},
    {'file': 'get_trading_signal_3094.py', 'name': '3094 聯傑'},
    {'file': 'get_trading_signal_3357.py', 'name': '3357 臺慶科'},
    {'file': 'get_trading_signal_3033.py', 'name': '3033 威健'},
    {'file': 'get_trading_signal_4933.py', 'name': '4933 友輝'},
    {'file': 'get_trading_signal_2472.py', 'name': '2472 立隆電'},
    {'file': 'get_trading_signal_8291.py', 'name': '8291 尚茂'},
    {'file': 'get_trading_signal_1710.py', 'name': '1710 東聯'},
    {'file': 'get_trading_signal_8473.py', 'name': '8473 山林水'},
    {'file': 'get_trading_signal_6451.py', 'name': '6451 訊芯-KY'},
    {'file': 'get_trading_signal_3023.py', 'name': '3023 信邦'},
    {'file': 'get_trading_signal_3167.py', 'name': '3167 大量'},
    {'file': 'get_trading_signal_2851.py', 'name': '2851 中再保'},
    {'file': 'get_trading_signal_1582.py', 'name': '1582 信錦'},
    {'file': 'get_trading_signal_6980.py', 'name': '6980 精成科'},
    {'file': 'get_trading_signal_2481.py', 'name': '2481 強茂'},
    {'file': 'get_trading_signal_2355.py', 'name': '2355 敬鵬'},
    {'file': 'get_trading_signal_7734.py', 'name': '7734 印能科技'},
    {'file': 'get_trading_signal_1595.py', 'name': '1595 川寶'},
    {'file': 'get_trading_signal_2442.py', 'name': '2442 新美齊'},
    {'file': 'get_trading_signal_1717.py', 'name': '1717 長興'},
    {'file': 'get_trading_signal_2417.py', 'name': '2417 圓剛'},
    {'file': 'get_trading_signal_2059.py', 'name': '2059 川湖'},
    {'file': 'get_trading_signal_1773.py', 'name': '1773 勝一'},
    {'file': 'get_trading_signal_3680.py', 'name': '3680 家登'},
    {'file': 'get_trading_signal_3498.py', 'name': '3498 陽程'},
    {'file': 'get_trading_signal_8028.py', 'name': '8028 昇陽半導體'},
    {'file': 'get_trading_signal_3587.py', 'name': '3587 閎康'},
    {'file': 'get_trading_signal_2340.py', 'name': '2340 台亞'},
    {'file': 'get_trading_signal_3714.py', 'name': '3714 富采'},
    {'file': 'get_trading_signal_3535.py', 'name': '3535 晶彩科'},
    {'file': 'get_trading_signal_3265.py', 'name': '3265 台星科'},
    {'file': 'get_trading_signal_3163.py', 'name': '3163 波若威'},
    {'file': 'get_trading_signal_3581.py', 'name': '3581 博磊'},
    {'file': 'get_trading_signal_4966.py', 'name': '4966 譜瑞-KY'},
    {'file': 'get_trading_signal_2426.py', 'name': '2426 鼎元'},
    {'file': 'get_trading_signal_4979.py', 'name': '4979 華星光'},
    {'file': 'get_trading_signal_3450.py', 'name': '3450 聯鈞'},
    {'file': 'get_trading_signal_3455.py', 'name': '3455 由田'},
    {'file': 'get_trading_signal_8064.py', 'name': '8064 東捷'},
    {'file': 'get_trading_signal_6257.py', 'name': '6257 矽格'},
    {'file': 'get_trading_signal_6147.py', 'name': '6147 頎邦'},
    {'file': 'get_trading_signal_3264.py', 'name': '3264 欣銓'},
    {'file': 'get_trading_signal_6138.py', 'name': '6138 茂達'},
    # ── added: scripts existed but were missing from this list ──
    {'file': 'get_trading_signal_1326.py', 'name': '1326 台化'},
    {'file': 'get_trading_signal_1425.py', 'name': '1425 台股'},
    {'file': 'get_trading_signal_2492.py', 'name': '2492 華新科'},
    {'file': 'get_trading_signal_3008.py', 'name': '3008 大立光'},
    {'file': 'get_trading_signal_3044.py', 'name': '3044 健鼎'},
    {'file': 'get_trading_signal_3260.py', 'name': '3260 威剛'},
    {'file': 'get_trading_signal_3576.py', 'name': '3576 聯合再生'},
    {'file': 'get_trading_signal_3665.py', 'name': '3665 貿聯-KY'},
    {'file': 'get_trading_signal_4720.py', 'name': '4720 德明'},
    {'file': 'get_trading_signal_4989.py', 'name': '4989 聯昌電子'},
    {'file': 'get_trading_signal_5274.py', 'name': '5274 信驊'},
    {'file': 'get_trading_signal_5483.py', 'name': '5483 中美矽晶'},
    {'file': 'get_trading_signal_1815.py', 'name': '1815 富喬'},
    {'file': 'get_trading_signal_6223.py', 'name': '6223 旺矽'},
    {'file': 'get_trading_signal_8358.py', 'name': '8358 金居'},
    {'file': 'get_trading_signal_6163.py', 'name': '6163 華電網'},
    {'file': 'get_trading_signal_6271.py', 'name': '6271 同欣電'},
    {'file': 'get_trading_signal_6274.py', 'name': '6274 台燿'},
    {'file': 'get_trading_signal_6485.py', 'name': '6485 幸康'},
    {'file': 'get_trading_signal_6488.py', 'name': '6488 環球晶'},
    {'file': 'get_trading_signal_7717.py', 'name': '7717 光鋐'},
    {'file': 'get_trading_signal_8438.py', 'name': '8438 艾鍗科技'},
]

# Signal emoji map for display
SIGNAL_EMOJI = {
    'BUY':  '🟢', 'SELL': '🔴', 'WAIT': '🟡',
    'HOLD': '🟡', '买入': '🟢', '卖出': '🔴',
    '观望': '🟡', '持有': '🟡',
}

def _parse_stdout(stdout):
    """
    Parse price, volume, signal and display symbol from a signal script's stdout.
    Returns dict with keys: symbol, price, volume, signal, money_amount, ai_accuracy
    """
    price   = None
    volume  = None
    signal  = '—'
    symbol  = None
    ai_acc  = '—'

    in_summary = False
    for line in stdout.split('\n'):
        stripped = line.strip()

        # ── enter summary block ──
        if '快速摘要' in stripped:
            in_summary = True
            continue

        if not in_summary:
            # price:   "   当前价格: NT$2635.00"  or  "   当前价格: $503.89"
            m = re.search(r'当前价格:\s*(?:NT\$|\$)([\d,]+\.?\d*)', stripped)
            if m:
                price = float(m.group(1).replace(',', ''))

            # volume:  "   今日成交量: 2,172,291"
            m = re.search(r'今日成交量:\s*([\d,]+)', stripped)
            if m:
                volume = int(m.group(1).replace(',', ''))

            # AI accuracy from header: "模型準確度: 🟢 AI準確度: 83.4/100 ..."
            m = re.search(r'AI準確度:\s*([\d.]+)/100', stripped)
            if m:
                ai_acc = m.group(1)

        else:
            # inside 快速摘要
            # "   股票: 2360.TW (致茂)"
            m = re.search(r'股票:\s*(\S+.*)', stripped)
            if m:
                symbol = m.group(1).strip()

            # "   信号: 买入 (BUY)"
            m = re.search(r'信号:\s*(.+)', stripped)
            if m:
                signal = m.group(1).strip()

            # accuracy may repeat in summary too
            m = re.search(r'AI準確度:\s*([\d.]+)/100', stripped)
            if m:
                ai_acc = m.group(1)

    money_amount = (price * volume) if (price and volume) else 0
    return {
        'symbol':       symbol or '—',
        'price':        price,
        'volume':       volume,
        'signal':       signal,
        'money_amount': money_amount,
        'ai_accuracy':  ai_acc,
    }


def run_signal(script_file, stock_name):
    """
    运行单个交易信号生成器。
    返回 (success, parsed_data, stdout_text)
    — stdout 不立即打印，由调用方统一输出。
    """
    try:
        script_dir  = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, script_file)

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=180,
            encoding='utf-8',
            errors='ignore',
            cwd=script_dir
        )

        if result.returncode == 0:
            parsed = _parse_stdout(result.stdout)
            if parsed['symbol'] == '—':
                parsed['symbol'] = stock_name
            return True, parsed, result.stdout
        else:
            return False, None, result.stderr

    except subprocess.TimeoutExpired:
        return False, None, '[X] 超时 (180秒)'
    except Exception as e:
        return False, None, f'[X] 错误: {e}'


def _fmt_money(amount):
    """Format money amount in NT$ with B/M/K suffix"""
    if amount >= 1_000_000_000:
        return f"NT${amount/1_000_000_000:.2f}B"
    elif amount >= 1_000_000:
        return f"NT${amount/1_000_000:.1f}M"
    elif amount >= 1_000:
        return f"NT${amount/1_000:.0f}K"
    else:
        return f"NT${amount:.0f}"


def print_money_ranking(results):
    """Print stocks sorted by volume × price (成交金額) descending."""
    ranked = sorted(results, key=lambda x: x['money_amount'], reverse=True)

    print("\n" + "=" * 100, flush=True)
    print("💰 成交金額排行榜  (volume × price，由大到小)", flush=True)
    print("=" * 100, flush=True)

    # header
    print(f"{'排名':>4}  {'股票':<28}  {'價格':>10}  {'成交量':>14}  {'成交金額':>14}  {'信號':<20}  {'AI準確度':>8}", flush=True)
    print("-" * 100, flush=True)

    buy_list  = []
    wait_list = []

    for rank, r in enumerate(ranked, 1):
        sig   = r['signal']
        emoji = '🟢' if 'BUY' in sig or '买入' in sig else \
                '🔴' if 'SELL' in sig or '卖出' in sig else '🟡'

        price_str  = f"NT${r['price']:,.1f}"  if r['price']  else '—'
        vol_str    = f"{r['volume']:,}"        if r['volume'] else '—'
        money_str  = _fmt_money(r['money_amount']) if r['money_amount'] else '—'
        acc_str    = f"{r['ai_accuracy']}/100" if r['ai_accuracy'] != '—' else '尚無數據'

        print(f"{rank:>4}  {r['symbol']:<28}  {price_str:>10}  {vol_str:>14}  {money_str:>14}  {emoji} {sig:<18}  {acc_str:>8}", flush=True)

        if 'BUY' in sig or '买入' in sig:
            buy_list.append(r['symbol'])
        elif 'WAIT' in sig or 'HOLD' in sig or '观望' in sig or '持有' in sig:
            wait_list.append(r['symbol'])

    print("=" * 100, flush=True)

    # Quick buy summary
    if buy_list:
        print(f"\n🟢 買入信號 ({len(buy_list)} 檔):", flush=True)
        for s in buy_list:
            # find money amount
            m = next((r['money_amount'] for r in ranked if r['symbol'] == s), 0)
            print(f"   • {s}  ({_fmt_money(m)})", flush=True)

    if wait_list:
        print(f"\n🟡 觀望/持有信號 ({len(wait_list)} 檔):", flush=True)
        for s in wait_list:
            m = next((r['money_amount'] for r in ranked if r['symbol'] == s), 0)
            print(f"   • {s}  ({_fmt_money(m)})", flush=True)

    print("", flush=True)


if __name__ == "__main__":
    total = len(SIGNAL_SCRIPTS)
    print("=" * 100, flush=True)
    print("批量运行所有台股交易信号生成器", flush=True)
    print("=" * 100, flush=True)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"总共 {total} 个台股  ── 收集完毕后先显示排行榜，再输出详细报告", flush=True)
    print("=" * 100, flush=True)

    success_count = 0
    failed_stocks = []
    all_results   = []   # (stock_name, stdout, parsed)

    # ── Phase 1: run all scripts, show one-line live progress ──
    for i, script in enumerate(SIGNAL_SCRIPTS, 1):
        name = script['name']
        print(f"  ⏳ [{i:>3}/{total}] {name:<30}", end='', flush=True)

        success, parsed, stdout = run_signal(script['file'], name)

        if success:
            success_count += 1
            sig   = parsed['signal']
            emoji = '🟢' if 'BUY' in sig or '买入' in sig else \
                    '🔴' if 'SELL' in sig or '卖出' in sig else '🟡'
            money = _fmt_money(parsed['money_amount']) if parsed['money_amount'] else '—'
            print(f"  {emoji} {sig:<22}  成交金額 {money}", flush=True)
            all_results.append({'name': name, 'stdout': stdout, 'parsed': parsed})
        else:
            print(f"  ❌ 失败", flush=True)
            failed_stocks.append(name)

    # ── Phase 2: ranking table (printed FIRST in the detailed section) ──
    parsed_list = [r['parsed'] for r in all_results if r['parsed']['money_amount'] > 0]
    if parsed_list:
        print_money_ranking(parsed_list)

    # ── Phase 3: full detailed reports for every stock ──
    print("\n" + "=" * 100, flush=True)
    print("📋 各股詳細報告", flush=True)
    print("=" * 100, flush=True)

    for r in all_results:
        print("\n" + "=" * 100, flush=True)
        print(f"運行: {r['name']}", flush=True)
        print("=" * 100, flush=True)
        print(r['stdout'], flush=True)

    # ── Phase 4: final summary ──
    print("\n" + "=" * 100, flush=True)
    print("批量运行完成!", flush=True)
    print("=" * 100, flush=True)
    print(f"成功运行: {success_count}/{total}", flush=True)
    print(f"失败数量: {len(failed_stocks)}", flush=True)

    if failed_stocks:
        print(f"\n失败的股票:", flush=True)
        for stock in failed_stocks:
            print(f"   - {stock}", flush=True)

    print("\n所有台股信号生成完成!", flush=True)
