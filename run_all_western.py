"""
批量运行所有全球股票的交易信号生成器 (US + EU + ASIA)
================================
自动运行所有已训练的美股、欧股和亚洲股票模型的交易信号
包括: 美国、欧洲、香港、日本、韩国
"""

import subprocess
import sys
import io
from datetime import datetime
import yfinance as yf

# 修复 Windows 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def get_peg_ratio(ticker_symbol):
    """获取 PEG 比率"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        # 尝试多个可能的键（US stocks 用 pegRatio，Taiwan/Asia stocks 用 trailingPegRatio）
        peg_ratio = info.get('pegRatio') or info.get('trailingPegRatio')
        return peg_ratio
    except Exception as e:
        return None

# 西方股票信号生成器 (US + EU)
SIGNAL_SCRIPTS = [
    # US Stocks 
    {'file': 'get_trading_signal_sndk.py', 'name': 'SNDK SanDisk'},
    {'file': 'get_trading_signal_tsm.py', 'name': 'TSM TSMC ADR'},
    {'file': 'get_trading_signal_wdc.py', 'name': 'WDC Western Digital'}, # Missing stock: WDC
    {'file': 'get_trading_signal_tsla.py', 'name': 'TSLA Tesla'},
    {'file': 'get_trading_signal_googl.py', 'name': 'GOOGL Alphabet'},
    {'file': 'get_trading_signal_amzn.py', 'name': 'AMZN Amazon'},
    {'file': 'get_trading_signal_mu.py', 'name': 'MU Micron'},
    {'file': 'get_trading_signal_amat.py', 'name': 'AMAT Applied Materials'},
    {'file': 'get_trading_signal_nvda.py', 'name': 'NVDA NVIDIA'},
    {'file': 'get_trading_signal_aapl.py', 'name': 'AAPL Apple'},
    {'file': 'get_trading_signal_aeva.py', 'name': 'AEVA Aeva Technologies Inc'},
    {'file': 'get_trading_signal_alab.py', 'name': 'ALAB Astera Labs Inc'},
    {'file': 'get_trading_signal_amkr.py', 'name': 'AMKR Amkor'},
    {'file': 'get_trading_signal_avav.py', 'name': 'AVAV AeroVironment'},
    {'file': 'get_trading_signal_avgo.py', 'name': 'AVGO Broadcom'},
    {'file': 'get_trading_signal_etn.py', 'name': 'ETN Eaton'},
    {'file': 'get_trading_signal_goog.py', 'name': 'GOOG Google'},
    {'file': 'get_trading_signal_htgc.py', 'name': 'HTGC Hercules Capital Inc'},   
    {'file': 'get_trading_signal_nat.py', 'name': 'NAT Nordic American Tankers'},
    {'file': 'get_trading_signal_nxpi.py', 'name': 'NXPI NXP Semiconductors'},
    {'file': 'get_trading_signal_oklo.py', 'name': 'OKLO Oklo Inc'},
    {'file': 'get_trading_signal_omer.py', 'name': 'OMER Omeros Corporation'},
    {'file': 'get_trading_signal_onds.py', 'name': 'ONDS Ondas Holdings'},
    {'file': 'get_trading_signal_pltr.py', 'name': 'PLTR Palantir'},
    {'file': 'get_trading_signal_rklb.py', 'name': 'RKLB Rocket Lab'},
    {'file': 'get_trading_signal_amd.py', 'name': 'AMD Advanced Micro Devices'},
    {'file': 'get_trading_signal_apld.py', 'name': 'APLD Applied Digital'},
    {'file': 'get_trading_signal_arm.py', 'name': 'ARM Arm Holdings'},
    {'file': 'get_trading_signal_crdo.py', 'name': 'CRDO Credo Technology'},
    {'file': 'get_trading_signal_fn.py', 'name': 'FN Fabrinet'},
    {'file': 'get_trading_signal_gild.py', 'name': 'GILD Gilead Sciences'},
    {'file': 'get_trading_signal_hsai.py', 'name': 'HSAI Hesai Group'},
    {'file': 'get_trading_signal_mrna.py', 'name': 'MRNA Moderna'},
    {'file': 'get_trading_signal_nem.py', 'name': 'NEM Newmont'},
    {'file': 'get_trading_signal_docn.py', 'name': 'DOCN Digital Ocean'},
    {'file': 'get_trading_signal_intc.py', 'name': 'INTC Intel'},
    {'file': 'get_trading_signal_invz.py', 'name': 'INVZ Innoviz Technologies'},
    {'file': 'get_trading_signal_ionq.py', 'name': 'IONQ IonQ'},
    {'file': 'get_trading_signal_klac.py', 'name': 'KLAC KLA Corporation'},
    {'file': 'get_trading_signal_nvo.py', 'name': 'NVO Novo Nordisk'},
    {'file': 'get_trading_signal_oust.py', 'name': 'OUST Ouster'},
    {'file': 'get_trading_signal_qubt.py', 'name': 'QUBT Quantum Computing Inc'},
    {'file': 'get_trading_signal_rdw.py', 'name': 'RDW Redwire'},
    {'file': 'get_trading_signal_rgti.py', 'name': 'RGTI Rigetti Computing'},
    {'file': 'get_trading_signal_satl.py', 'name': 'SATL Satellogic'},
    {'file': 'get_trading_signal_smci.py', 'name': 'SMCI Super Micro Computer'},
    {'file': 'get_trading_signal_smr.py', 'name': 'SMR NuScale Power'},
    {'file': 'get_trading_signal_snow.py', 'name': 'SNOW Snowflake'},
    {'file': 'get_trading_signal_vrt.py', 'name': 'VRT Vertiv Holdings'},
    {'file': 'get_trading_signal_lite.py', 'name': 'LITE Lumentum'},

    # European Stocks
    {'file': 'get_trading_signal_rnmby.py', 'name': 'RNMBY Rheinmetall AG'},
    {'file': 'get_trading_signal_rhm.py', 'name': 'RHM.DE Rheinmetall'},

    # Hong Kong Stocks
    {'file': 'get_trading_signal_02202.py', 'name': '02202.HK Vanke'},
    {'file': 'get_trading_signal_01810.py', 'name': '01810.HK Xiaomi'},

    # Japan Stocks
    {'file': 'get_trading_signal_9984.py', 'name': '9984.T SoftBank Group'},

    # South Korea Stocks - Defense
    {'file': 'get_trading_signal_079550.py', 'name': '079550.KS LIG Nex1'},
    {'file': 'get_trading_signal_012450.py', 'name': '012450.KS Hanwha Aerospace'},

    # US Stocks - Additional
    {'file': 'get_trading_signal_orcl.py', 'name': 'ORCL Oracle Corporation'},
    {'file': 'get_trading_signal_mdb.py', 'name': 'MDB MongoDB'},
    {'file': 'get_trading_signal_ddog.py', 'name': 'DDOG Datadog'},
    {'file': 'get_trading_signal_meta.py', 'name': 'META Meta Platforms'},
    {'file': 'get_trading_signal_stx.py', 'name': 'STX Seagate Technology'},
    {'file': 'get_trading_signal_mpwr.py', 'name': 'MPWR Monolithic Power Systems'},
    {'file': 'get_trading_signal_deck.py', 'name': 'DECK Deckers Outdoor'},
    {'file': 'get_trading_signal_txn.py', 'name': 'TXN Texas Instruments'},
    {'file': 'get_trading_signal_mrvl.py', 'name': 'MRVL Marvell Technology'},
    {'file': 'get_trading_signal_snps.py', 'name': 'SNPS Synopsys'},
    {'file': 'get_trading_signal_cdns.py', 'name': 'CDNS Cadence Design Systems'},
    {'file': 'get_trading_signal_msft.py', 'name': 'MSFT Microsoft'},
    {'file': 'get_trading_signal_omc.py', 'name': 'OMC Omnicom'},
    {'file': 'get_trading_signal_grmn.py', 'name': 'GRMN Garmin'},
    {'file': 'get_trading_signal_gpn.py', 'name': 'GPN Global Payments'},
    {'file': 'get_trading_signal_bax.py', 'name': 'BAX Baxter'},
    {'file': 'get_trading_signal_moh.py', 'name': 'MOH Molina Healthcare'},
    {'file': 'get_trading_signal_tpl.py', 'name': 'TPL Texas Pacific Land'},
    {'file': 'get_trading_signal_coin.py', 'name': 'COIN Coinbase'},
    {'file': 'get_trading_signal_cien.py', 'name': 'CIEN Ciena'},

]

def run_signal(script_file, stock_name, peg_ratio=None):
    """运行单个交易信号生成器"""
    print("\n" + "=" * 100)
    print(f"运行: {stock_name}")
    if peg_ratio is not None and peg_ratio > 0:
        print(f"PEG 比率: {peg_ratio:.2f}")
    else:
        print(f"PEG 比率: N/A")
    print("=" * 100)

    try:
        import os
        # 获取当前脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, script_file)

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=180,  # Increased to 3 minutes for FinBERT loading
            encoding='utf-8',
            errors='ignore',
            cwd=script_dir  # 设置工作目录
        )

        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"[X] 运行失败:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"[X] 超时 (180秒)")
        return False
    except Exception as e:
        print(f"[X] 错误: {e}")
        return False

if __name__ == "__main__":
    print("=" * 100)
    print("批量运行所有西方股票交易信号生成器 (US + EU)")
    print("=" * 100)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总共 {len(SIGNAL_SCRIPTS)} 个股票 (US: 45, EU: 2, HK: 2)")
    print("=" * 100)

    success_count = 0
    failed_stocks = []
    peg_ratios = {}  # 存储 PEG 比率

    for i, script in enumerate(SIGNAL_SCRIPTS, 1):
        print(f"\n进度: [{i}/{len(SIGNAL_SCRIPTS)}]")

        # 提取股票代碼 (取名稱第一個單詞)
        ticker_symbol = script['name'].split()[0]

        # 獲取 PEG 比率
        peg_ratio = get_peg_ratio(ticker_symbol)
        peg_ratios[script['name']] = peg_ratio

        success = run_signal(script['file'], script['name'], peg_ratio)

        if success:
            success_count += 1
        else:
            failed_stocks.append(script['name'])

        # Add a small delay to avoid API rate limits
        if i < len(SIGNAL_SCRIPTS):
            import time
            time.sleep(2)  # 2 second delay between requests

    # 最终总结
    print("\n" + "=" * 100)
    print("批量运行完成!")
    print("=" * 100)
    print(f"成功运行: {success_count}/{len(SIGNAL_SCRIPTS)}")
    print(f"失败数量: {len(failed_stocks)}")

    if failed_stocks:
        print(f"\n失败的股票:")
        for stock in failed_stocks:
            print(f"   - {stock}")

    # 顯示 PEG 比率總結
    print("\n" + "=" * 100)
    print("PEG 比率總結 (Price/Earnings to Growth Ratio)")
    print("=" * 100)

    # 過濾出有效的 PEG 比率並排序
    valid_pegs = [(name, peg) for name, peg in peg_ratios.items() if peg is not None and peg > 0]
    valid_pegs.sort(key=lambda x: x[1])  # 按 PEG 比率排序

    if valid_pegs:
        print(f"\n{'股票':<40} {'PEG 比率':>10}")
        print("-" * 52)
        for name, peg in valid_pegs:
            peg_status = "低估" if peg < 1.0 else "合理" if peg < 2.0 else "高估"
            print(f"{name:<40} {peg:>10.2f}  ({peg_status})")

        avg_peg = sum(peg for _, peg in valid_pegs) / len(valid_pegs)
        print("-" * 52)
        print(f"{'平均 PEG 比率:':<40} {avg_peg:>10.2f}")
        print(f"\n💡 PEG < 1.0: 可能被低估")
        print(f"💡 PEG 1.0-2.0: 估值合理")
        print(f"💡 PEG > 2.0: 可能被高估")

    na_count = len([p for p in peg_ratios.values() if p is None or p <= 0])
    if na_count > 0:
        print(f"\n⚠️  {na_count} 支股票無 PEG 比率數據")

    print("\n所有西方股票信号生成完成!")
