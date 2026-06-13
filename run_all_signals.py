"""
批量运行所有股票的交易信号生成器
================================
自动运行所有已训练模型的交易信号
"""

import subprocess
import sys
import io
from datetime import datetime

# 修复 Windows 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 已有信号生成器的股票
SIGNAL_SCRIPTS = [
    # US Stocks
    {'file': 'get_trading_signal_aapl.py', 'name': 'AAPL Apple'},
    {'file': 'get_trading_signal_avgo.py', 'name': 'AVGO Broadcom'},
    {'file': 'get_trading_signal_goog.py', 'name': 'GOOG Google'},
    {'file': 'get_trading_signal_mu.py', 'name': 'MU Micron'},
    {'file': 'get_trading_signal_nvda.py', 'name': 'NVDA NVIDIA'},
    {'file': 'get_trading_signal_omer.py', 'name': 'OMER Omeros Corporation'},
    {'file': 'get_trading_signal_alab.py', 'name': 'ALAB Astera Labs Inc'},
    {'file': 'get_trading_signal_nat.py', 'name': 'NAT Nordic American Tankers'},
    {'file': 'get_trading_signal_htgc.py', 'name': 'HTGC Hercules Capital Inc'},

    # European Stocks
    {'file': 'get_trading_signal_rhm.py', 'name': 'RHM.DE Rheinmetall AG'},

    # Taiwan Stocks
    {'file': 'get_trading_signal_1519.py', 'name': '1519 台股'},
    {'file': 'get_trading_signal_2317.py', 'name': '2317 鴻海'},
    {'file': 'get_trading_signal_2330.py', 'name': '2330 台積電'},
    {'file': 'get_trading_signal_2337.py', 'name': '2337 旺宏'},
    {'file': 'get_trading_signal_2344.py', 'name': '2344 華邦電'},
    {'file': 'get_trading_signal_2360.py', 'name': '2360 致茂'},
    {'file': 'get_trading_signal_2360.py', 'name': '2360 致茂'},
    {'file': 'get_trading_signal_2408.py', 'name': '2383 台光電'},
    {'file': 'get_trading_signal_2449.py', 'name': '2449 台股'},
    {'file': 'get_trading_signal_2451.py', 'name': '2451 創見資訊'},
    {'file': 'get_trading_signal_3017.py', 'name': '3017 台股'},
    {'file': 'get_trading_signal_3653.py', 'name': '3653 健策'},
    {'file': 'get_trading_signal_3661.py', 'name': '3661 世芯-KY'},
    {'file': 'get_trading_signal_3711.py', 'name': '3711 台股'},
    {'file': 'get_trading_signal_3715.py', 'name': '3715 台股'},
    {'file': 'get_trading_signal_4938.py', 'name': '4938 台股'},
    {'file': 'get_trading_signal_6209.py', 'name': '6209 台股'},
    {'file': 'get_trading_signal_6269.py', 'name': '6269 台股'},
    {'file': 'get_trading_signal_6442.py', 'name': '6442 兆豐金控'},
    {'file': 'get_trading_signal_6443.py', 'name': '6443 台股'},
    {'file': 'get_trading_signal_6515.py', 'name': '6515 台股'},
    {'file': 'get_trading_signal_6770.py', 'name': '6770 力積電'},
    {'file': 'get_trading_signal_6781.py', 'name': '6781 AES-KY'},
    {'file': 'get_trading_signal_6805.py', 'name': '6805 台股'},
    {'file': 'get_trading_signal_7769.py', 'name': '7769 霖揚'},
    {'file': 'get_trading_signal_8131.py', 'name': '8131 福懋科'},
    {'file': 'get_trading_signal_8210.py', 'name': '8210 勤誠'},
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
            timeout=60,
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
        print(f"[X] 超时 (60秒)")
        return False
    except Exception as e:
        print(f"[X] 错误: {e}")
        return False

if __name__ == "__main__":
    print("=" * 100)
    print("批量运行所有股票交易信号生成器")
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

    print("\n所有信号生成完成!")
