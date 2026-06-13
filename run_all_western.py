"""
批量运行所有西方股票的交易信号生成器 (US + EU)
================================
自动运行所有已训练的美股和欧股模型的交易信号
+ 整合爆量後續分析策略
"""

import subprocess
import sys
import io
from datetime import datetime

# 修复 Windows 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 導入爆量分析模組
try:
    from simu_deepseek_trading_signal_goog import run_batch_analysis
    SURGE_ANALYSIS_AVAILABLE = True
except ImportError:
    SURGE_ANALYSIS_AVAILABLE = False
    run_batch_analysis = None

# 西方股票信号生成器 (US + EU)
SIGNAL_SCRIPTS = [
    # US Stocks
    {'file': 'get_trading_signal_aapl.py', 'name': 'AAPL Apple'},
    {'file': 'get_trading_signal_aeva.py', 'name': 'AEVA Aeva Technologies Inc'},
    {'file': 'get_trading_signal_alab.py', 'name': 'ALAB Astera Labs Inc'},
    {'file': 'get_trading_signal_amkr.py', 'name': 'AMKR Amkor'},
    {'file': 'get_trading_signal_avav.py', 'name': 'AVAV AeroVironment'},
    {'file': 'get_trading_signal_avgo.py', 'name': 'AVGO Broadcom'},
    {'file': 'get_trading_signal_etn.py', 'name': 'ETN Eaton'},
    {'file': 'get_trading_signal_goog.py', 'name': 'GOOG Google'},
    {'file': 'get_trading_signal_htgc.py', 'name': 'HTGC Hercules Capital Inc'},
    {'file': 'get_trading_signal_mu.py', 'name': 'MU Micron'},
    {'file': 'get_trading_signal_nat.py', 'name': 'NAT Nordic American Tankers'},
    {'file': 'get_trading_signal_nvda.py', 'name': 'NVDA NVIDIA'},
    {'file': 'get_trading_signal_nxpi.py', 'name': 'NXPI NXP Semiconductors'},
    {'file': 'get_trading_signal_oklo.py', 'name': 'OKLO Oklo Inc'},
    {'file': 'get_trading_signal_omer.py', 'name': 'OMER Omeros Corporation'},
    {'file': 'get_trading_signal_onds.py', 'name': 'ONDS Ondas Holdings'},
    {'file': 'get_trading_signal_pltr.py', 'name': 'PLTR Palantir'},
    {'file': 'get_trading_signal_rklb.py', 'name': 'RKLB Rocket Lab'},
    {'file': 'get_trading_signal_tsm.py', 'name': 'TSM TSMC ADR'},
    {'file': 'get_trading_signal_wdc.py', 'name': 'WDC Western Digital'},
    {'file': 'get_trading_signal_stx.py', 'name': 'STX Seagate'},
    {'file': 'get_trading_signal_amd.py', 'name': 'AMD Advanced Micro Devices'},
    {'file': 'get_trading_signal_intc.py', 'name': 'INTC Intel'},
    {'file': 'get_trading_signal_mchp.py', 'name': 'MCHP Microchip Technology'},
    {'file': 'get_trading_signal_snps.py', 'name': 'SNPS Synopsys'},
    {'file': 'get_trading_signal_mpwr.py', 'name': 'MPWR Monolithic Power Systems'},
    {'file': 'get_trading_signal_txn.py', 'name': 'TXN Texas Instruments'},
    {'file': 'get_trading_signal_on.py', 'name': 'ON ON Semiconductor'},
    {'file': 'get_trading_signal_arm.py', 'name': 'ARM Arm Holdings'},
    {'file': 'get_trading_signal_mrvl.py', 'name': 'MRVL Marvell Technology'},
    {'file': 'get_trading_signal_bkr.py', 'name': 'BKR Baker Hughes'},
    {'file': 'get_trading_signal_uri.py', 'name': 'URI United Rentals'},
    {'file': 'get_trading_signal_gev.py', 'name': 'GEV GE Vernova'},
    {'file': 'get_trading_signal_stld.py', 'name': 'STLD Steel Dynamics'},
    {'file': 'get_trading_signal_klac.py', 'name': 'KLAC KLA Corporation'},
    {'file': 'get_trading_signal_crdo.py', 'name': 'CRDO Credo Technology'},
    {'file': 'get_trading_signal_tsla.py', 'name': 'TSLA Tesla'},
    {'file': 'get_trading_signal_ionq.py', 'name': 'IONQ'},
    {'file': 'get_trading_signal_QBTS.py', 'name': 'QBTS Quantum Computing'},
    {'file': 'get_trading_signal_RGTI.py', 'name': 'RGTI Rigetti Computing'},
    {'file': 'get_trading_signal_9888.py', 'name': 'Baidu ADR'},
    {'file': 'get_trading_signal_301005.py', 'name': 'essence fastening'},
    {'file': 'get_trading_signal_sndk.py', 'name': 'sndk'},
    {'file': 'get_trading_signal_apld.py', 'name': 'apld'},
    {'file': 'get_trading_signal_orcl.py', 'name': 'ORCL Oracle Corporation'},
    {'file': 'get_trading_signal_qcom.py', 'name': 'QCOM Qualcomm'},
    {'file': 'get_trading_signal_oust.py', 'name': 'OUST Ouster Inc'},
    {'file': 'get_trading_signal_smci.py', 'name': 'SMCI Super Micro Computer'},
    {'file': 'get_trading_signal_ibm.py', 'name': 'IBM IBM Corporation'},
    {'file': 'get_trading_signal_amzn.py', 'name': 'AMZN Amazon'},
    {'file': 'get_trading_signal_ba.py', 'name': 'BA Boeing'},
    {'file': 'get_trading_signal_dell.py', 'name': 'DELL Dell Technologies'},
    {'file': 'get_trading_signal_crwd.py', 'name': 'CRWD CrowdStrike'},
    {'file': 'get_trading_signal_ctsh.py', 'name': 'CTSH Cognizant'},
    {'file': 'get_trading_signal_dxcm.py', 'name': 'DXCM Dexcom'},
    {'file': 'get_trading_signal_hpq.py', 'name': 'HPQ HP Inc'},
    {'file': 'get_trading_signal_swks.py', 'name': 'SWKS Skyworks Solutions'},
    {'file': 'get_trading_signal_ntap.py', 'name': 'NTAP NetApp'},
    {'file': 'get_trading_signal_rl.py', 'name': 'RL Ralph Lauren'},
    {'file': 'get_trading_signal_rost.py', 'name': 'ROST Ross Stores'},
    {'file': 'get_trading_signal_wsm.py', 'name': 'WSM Williams-Sonoma'},
    {'file': 'get_trading_signal_zs.py', 'name': 'ZS Zscaler'},
    {'file': 'get_trading_signal_aaoi.py', 'name': 'AAOI Applied Optoelectronics'},
    {'file': 'get_trading_signal_mdb.py', 'name': 'MDB MongoDB'},
    {'file': 'get_trading_signal_docn.py', 'name': 'DOCN DigitalOcean'},
    {'file': 'get_trading_signal_brk-a.py', 'name': 'BRK-A Berkshire Hathaway A'},
    {'file': 'get_trading_signal_brk-b.py', 'name': 'BRK-B Berkshire Hathaway B'},
    {'file': 'get_trading_signal_lulu.py', 'name': 'LULU Lululemon'},
    {'file': 'get_trading_signal_coin.py', 'name': 'COIN Coinbase'},
    {'file': 'get_trading_signal_crm.py', 'name': 'CRM Salesforce'},
    {'file': 'get_trading_signal_mcd.py', 'name': 'MCD McDonald\'s'},
    {'file': 'get_trading_signal_shop.py', 'name': 'SHOP Shopify'},
    {'file': 'get_trading_signal_rddt.py', 'name': 'RDDT Reddit'},
    {'file': 'get_trading_signal_dis.py', 'name': 'DIS Walt Disney'},
    {'file': 'get_trading_signal_ko.py', 'name': 'KO Coca-Cola'},
    {'file': 'get_trading_signal_lin.py', 'name': 'LIN Linde'},
    {'file': 'get_trading_signal_lite.py', 'name': 'LITE Lumentum'},

    # European Stocks
    {'file': 'get_trading_signal_rnmby.py', 'name': 'RNMBY Rheinmetall AG'},
]

def run_signal(script_file, stock_name):
    """运行单个交易信号生成器"""
    print("\n" + "=" * 100)
    print(f"运行: {stock_name}")
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
    print(f"总共 {len(SIGNAL_SCRIPTS)} 个股票")
    print("=" * 100)

    success_count = 0
    failed_stocks = []

    for i, script in enumerate(SIGNAL_SCRIPTS, 1):
        print(f"\n进度: [{i}/{len(SIGNAL_SCRIPTS)}]")

        success = run_signal(script['file'], script['name'])

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

    # ================================================================
    # 爆量後續分析 (風險評估)
    # ================================================================
    if SURGE_ANALYSIS_AVAILABLE:
        print("\n" + "=" * 100)
        print("爆量後續分析 - 風險評估")
        print("=" * 100)

        # 從 SIGNAL_SCRIPTS 提取 ticker
        tickers = []
        for script in SIGNAL_SCRIPTS:
            # 從檔名提取 ticker (e.g., get_trading_signal_goog.py -> GOOG)
            filename = script['file']
            if filename.startswith('get_trading_signal_') and filename.endswith('.py'):
                ticker = filename.replace('get_trading_signal_', '').replace('.py', '').upper()
                # 特殊處理
                if ticker.isdigit():
                    continue  # 跳過純數字 (中國股票代碼)
                if ticker == '9888':
                    ticker = 'BIDU'  # Baidu ADR
                elif ticker == '301005':
                    continue  # 跳過 A 股
                tickers.append(ticker)

        # 執行爆量分析
        if tickers:
            surge_results = run_batch_analysis(tickers)

            # 輸出風險摘要
            print("\n" + "=" * 100)
            print("風險等級摘要")
            print("=" * 100)

            high_risk = [r for r in surge_results if r.get('risk_level') == 'HIGH' and r.get('success')]
            medium_risk = [r for r in surge_results if r.get('risk_level') == 'MEDIUM' and r.get('success')]
            low_risk = [r for r in surge_results if r.get('risk_level') == 'LOW' and r.get('success')]

            if high_risk:
                print(f"\n[!] 高風險股票 ({len(high_risk)} 檔) - 建議謹慎:")
                for r in high_risk:
                    print(f"    {r['ticker']}: 跌幅 {r['price_change']:.2f}%, 量比 {r['volume_ratio']:.2f}x")

            if medium_risk:
                print(f"\n[~] 中風險股票 ({len(medium_risk)} 檔) - 需要觀察:")
                for r in medium_risk:
                    print(f"    {r['ticker']}: 跌幅 {r['price_change']:.2f}%, 量比 {r['volume_ratio']:.2f}x")

            if low_risk:
                print(f"\n[V] 低風險股票 ({len(low_risk)} 檔) - 相對安全:")
                for r in low_risk:
                    print(f"    {r['ticker']}: 漲跌 {r['price_change']:.2f}%, 量比 {r['volume_ratio']:.2f}x")
    else:
        print("\n[!] 爆量分析模組未載入，跳過風險評估")

    print("\n" + "=" * 100)
    print("所有西方股票信号生成完成!")
    print("=" * 100)
