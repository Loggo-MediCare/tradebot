"""
批量运行所有台股的交易信号生成器
================================
自动运行所有已训练的台股模型的交易信号
"""

import subprocess
import sys
import io
import os
from datetime import datetime

# 修复 Windows 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 强制输出刷新（避免缓冲导致终端只显示最后几个股票）
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUNBUFFERED'] = '1'

# 台股信号生成器 (updated 2025-01)
SIGNAL_SCRIPTS = [
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
    {'file': 'get_trading_signal_2308.py', 'name': '2308 台達電'},
    {'file': 'get_trading_signal_2313.py', 'name': '2313 華通'},
    {'file': 'get_trading_signal_2317.py', 'name': '2317 鴻海'},
    {'file': 'get_trading_signal_2324.py', 'name': '2324 仁寶'},
    {'file': 'get_trading_signal_2327.py', 'name': '2327 國巨'},
    {'file': 'get_trading_signal_2330.py', 'name': '2330 台積電'},
    {'file': 'get_trading_signal_2331.py', 'name': '2331 精英'},
    {'file': 'get_trading_signal_2337.py', 'name': '2337 旺宏'},
    {'file': 'get_trading_signal_2344.py', 'name': '2344 華邦電'},
    {'file': 'get_trading_signal_2345.py', 'name': '2345 智邦'},
    {'file': 'get_trading_signal_2360.py', 'name': '2360 致茂'},
    {'file': 'get_trading_signal_2363.py', 'name': '2363 矽統'},
    {'file': 'get_trading_signal_2367.py', 'name': '2367 燿華'},
    {'file': 'get_trading_signal_2368.py', 'name': '2368 金像電'},
    {'file': 'get_trading_signal_2376.py', 'name': '2376 技嘉'},
    {'file': 'get_trading_signal_2382.py', 'name': '2382 廣達'},
    {'file': 'get_trading_signal_2383.py', 'name': '2383 台光電'},
    {'file': 'get_trading_signal_2385.py', 'name': '2385 群光'},
    {'file': 'get_trading_signal_2408.py', 'name': '2408 南亞科'},
    {'file': 'get_trading_signal_2409.py', 'name': '2409 友達'},
    {'file': 'get_trading_signal_2449.py', 'name': '2449 京元電子'},
    {'file': 'get_trading_signal_2451.py', 'name': '2451 創見資訊'},
    {'file': 'get_trading_signal_2454.py', 'name': '2454 聯發科'},
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
    {'file': 'get_trading_signal_3711.py', 'name': '3711 日月光投控'},
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
]

def run_signal(script_file, stock_name):
    """运行单个交易信号生成器"""
    print("\n" + "=" * 100, flush=True)
    print(f"运行: {stock_name}", flush=True)
    print("=" * 100, flush=True)

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
            print(result.stdout, flush=True)  # 强制刷新输出缓冲
            return True
        else:
            print(f"[X] 运行失败:", flush=True)
            print(result.stderr, flush=True)
            return False

    except subprocess.TimeoutExpired:
        print(f"[X] 超时 (180秒)")
        return False
    except Exception as e:
        print(f"[X] 错误: {e}")
        return False

if __name__ == "__main__":
    print("=" * 100, flush=True)
    print("批量运行所有台股交易信号生成器", flush=True)
    print("=" * 100, flush=True)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"总共 {len(SIGNAL_SCRIPTS)} 个台股", flush=True)
    print("=" * 100, flush=True)

    success_count = 0
    failed_stocks = []

    for i, script in enumerate(SIGNAL_SCRIPTS, 1):
        print(f"\n进度: [{i}/{len(SIGNAL_SCRIPTS)}]", flush=True)

        success = run_signal(script['file'], script['name'])

        if success:
            success_count += 1
        else:
            failed_stocks.append(script['name'])

    # 最终总结
    print("\n" + "=" * 100, flush=True)
    print("批量运行完成!", flush=True)
    print("=" * 100, flush=True)
    print(f"成功运行: {success_count}/{len(SIGNAL_SCRIPTS)}", flush=True)
    print(f"失败数量: {len(failed_stocks)}", flush=True)

    if failed_stocks:
        print(f"\n失败的股票:", flush=True)
        for stock in failed_stocks:
            print(f"   - {stock}", flush=True)

    print("\n所有台股信号生成完成!", flush=True)
