"""
批量运行所有台股的交易信号生成器 (输出到TXT文件)
================================
自动运行所有已训练的台股模型的交易信号
输出保存到单个文本文件

最新添加的XGBoost模型股票 (2026-03-02):
==========================================
Stock      | Name           | Accuracy | Model Type | Status
-----------|----------------|----------|------------|--------
3563.TW    | 牧德           | 60.28%   | XGBoost    | ✅
3576.TW    | 聯合再生       | 68.15%   | XGBoost    | ✅
3615.TWO   | 安可           | 67.54%   | XGBoost    | ✅
3665.TW    | 貿聯-KY        | 52.82%   | XGBoost    | ✅
4564.TW    | 元翎           | 65.32%   | XGBoost    | ✅
4577.TWO   | 達航科技       | 51.42%   | XGBoost    | ✅
4768.TWO   | 晶呈科技       | 50.97%   | XGBoost    | ✅
4989.TW    | 榮科           | 64.24%   | XGBoost    | ✅
4991.TWO   | 環宇-KY        | 53.83%   | XGBoost    | ✅
6220.TWO   | 岳豐           | 75.60%   | XGBoost    | 🌟 EXCELLENT
6230.TW    | 尼得科超眾     | 65.93%   | XGBoost    | ✅
6442.TW    | 光聖           | 50.00%   | XGBoost    | ✅
6526.TW    | 達發           | 49.28%   | XGBoost    | ⚠️
6789.TW    | 采鈺           | 56.77%   | XGBoost    | ✅
6830.TW    | 汎銓           | 62.50%   | XGBoost    | ✅
6877.TWO   | 鏵友益         | 70.75%   | XGBoost    | 🌟 EXCELLENT
8438.TW    | 昶昕           | 53.97%   | XGBoost    | ✅
8927.TWO   | 北基           | 67.74%   | XGBoost    | ✅

Top Performers (≥70%):
- 6220.TWO 岳豐: 75.60%
- 6877.TWO 鏵友益: 70.75%

All models trained: 2026-03-02
All signal files created: 2026-03-02
Added to run_all: 2026-03-02
"""

import subprocess
import sys
import io
import os
from datetime import datetime

# 修复 Windows 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 强制输出刷新
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUNBUFFERED'] = '1'

# 台股信号生成器
SIGNAL_SCRIPTS = [
    {'file': 'get_trading_signal_2337.py', 'name': '2337 旺宏'},
    {'file': 'get_trading_signal_3006.py', 'name': '3006 晶豪科'},
    {'file': 'get_trading_signal_2367.py', 'name': '2367 燿華'},
    {'file': 'get_trading_signal_6285.py', 'name': '6285 啟碁'},
    {'file': 'get_trading_signal_2313.py', 'name': '2313 華通'},
    {'file': 'get_trading_signal_2303.py', 'name': '2303 聯電'},
    {'file': 'get_trading_signal_8069.py', 'name': '8069 元太'},
    {'file': 'get_trading_signal_2308.py', 'name': '2308 台達電'},

    {'file': 'get_trading_signal_6683.py', 'name': '6683 雍智科技'},
    {'file': 'get_trading_signal_6223.py', 'name': '6223 旺矽'},
    {'file': 'get_trading_signal_3044.py', 'name': '3044 健鼎'},
    {'file': 'get_trading_signal_2891.py', 'name': '2891 中信金'},
    {'file': 'get_trading_signal_2002.py', 'name': '2002 中鋼'},

    {'file': 'get_trading_signal_6770.py', 'name': '6770 力積電'},
    {'file': 'get_trading_signal_6442.py', 'name': '6442 光聖'},
    {'file': 'get_trading_signal_1301.py', 'name': '1301 台塑'},
    {'file': 'get_trading_signal_3037.py', 'name': '3037 欣興'},

    {'file': 'get_trading_signal_1101.py', 'name': '1101 台泥'},
    {'file': 'get_trading_signal_1303.py', 'name': '1303 南亞'},
    {'file': 'get_trading_signal_1519.py', 'name': '1519 台股'},
    {'file': 'get_trading_signal_1605.py', 'name': '1605 華新'},
     {'file': 'get_trading_signal_2317.py', 'name': '2317 鴻海'},
    {'file': 'get_trading_signal_2327.py', 'name': '2327 國巨'},
    {'file': 'get_trading_signal_2330.py', 'name': '2330 台積電'},
    {'file': 'get_trading_signal_2344.py', 'name': '2344 華邦電'},
    {'file': 'get_trading_signal_2345.py', 'name': '2345 台股'},
    {'file': 'get_trading_signal_2360.py', 'name': '2360 致茂'},
    {'file': 'get_trading_signal_2368.py', 'name': '2368 金像電'},
    {'file': 'get_trading_signal_2376.py', 'name': '2376 台股'},
    {'file': 'get_trading_signal_2382.py', 'name': '2382 廣達'},
    {'file': 'get_trading_signal_2383.py', 'name': '2383 台光電'},
    {'file': 'get_trading_signal_2385.py', 'name': '2385 台股'},
    {'file': 'get_trading_signal_2408.py', 'name': '2408 南亞科'},
    {'file': 'get_trading_signal_2409.py', 'name': '2409 友達'},
    {'file': 'get_trading_signal_2449.py', 'name': '2449 京元電子'},
    {'file': 'get_trading_signal_2451.py', 'name': '2451 創見資訊'},
    {'file': 'get_trading_signal_2454.py', 'name': '2454 聯發科'},
    {'file': 'get_trading_signal_2603.py', 'name': '2603 長榮'},
    {'file': 'get_trading_signal_2884.py', 'name': '2884 玉山金'},
    {'file': 'get_trading_signal_2890.py', 'name': '2890 永豐金'},

    {'file': 'get_trading_signal_3030.py', 'name': '3030 德律'},
    {'file': 'get_trading_signal_3044.py', 'name': '3044 健鼎'},
    {'file': 'get_trading_signal_3017.py', 'name': '3017 台股'},
    {'file': 'get_trading_signal_3231.py', 'name': '3231 緯創'},
    {'file': 'get_trading_signal_3363.py', 'name': '3363 上詮'},
    {'file': 'get_trading_signal_3443.py', 'name': '3443 創意'},
    {'file': 'get_trading_signal_3449.py', 'name': '3449 鈺德'},
    {'file': 'get_trading_signal_3481.py', 'name': '3481 群創'},
    {'file': 'get_trading_signal_3653.py', 'name': '3653 健策'},
    {'file': 'get_trading_signal_3661.py', 'name': '3661 世芯-KY'},
    {'file': 'get_trading_signal_3711.py', 'name': '3711 日月光投控'},
    {'file': 'get_trading_signal_3715.py', 'name': '3715 台股'},
    {'file': 'get_trading_signal_4540.py', 'name': '4540 全球傳動'},
    {'file': 'get_trading_signal_4722.py', 'name': '4722 国精化'},
    {'file': 'get_trading_signal_4746.py', 'name': '4746 台股'},
    {'file': 'get_trading_signal_4938.py', 'name': '4938 台股'},
    {'file': 'get_trading_signal_5483.py', 'name': '5483 台股'},
    {'file': 'get_trading_signal_6209.py', 'name': '6209 台股'},
    {'file': 'get_trading_signal_6163.py', 'name': '6163 華電網'},
    {'file': 'get_trading_signal_7709.py', 'name': '7709 榮田'},
    {'file': 'get_trading_signal_6239.py', 'name': '6239 力成'},
    {'file': 'get_trading_signal_6269.py', 'name': '6269 台股'},

    {'file': 'get_trading_signal_6443.py', 'name': '6443 台股'},
    {'file': 'get_trading_signal_6472.py', 'name': '6472 保瑞'},
    {'file': 'get_trading_signal_6477.py', 'name': '6477 安集'},
    {'file': 'get_trading_signal_6223.py', 'name': '6223 旺矽'},
    {'file': 'get_trading_signal_6683.py', 'name': '6683 雍智科技'},
    {'file': 'get_trading_signal_6515.py', 'name': '6515 台股'},
    {'file': 'get_trading_signal_6669.py', 'name': '6669 緯穎'},

    {'file': 'get_trading_signal_6781.py', 'name': '6781 AES-KY'},
    {'file': 'get_trading_signal_6805.py', 'name': '6805 台股'},
    {'file': 'get_trading_signal_7717.py', 'name': '7717 萊德光電'},
    {'file': 'get_trading_signal_8021.py', 'name': '8021 尖點'},
    {'file': 'get_trading_signal_8110.py', 'name': '8110 台股'},
    {'file': 'get_trading_signal_8131.py', 'name': '8131 福懋科'},
    {'file': 'get_trading_signal_8150.py', 'name': '8150 南茂'},
    {'file': 'get_trading_signal_8210.py', 'name': '8210 勤誠'},
    {'file': 'get_trading_signal_8499.py', 'name': '8499 鼎炫-KY'},
    {'file': 'get_trading_signal_2357.py', 'name': '2357 華碩'},
    {'file': 'get_trading_signal_2363.py', 'name': '2363 矽統'},
    {'file': 'get_trading_signal_2634.py', 'name': '2634 漢翔'},
    {'file': 'get_trading_signal_3004.py', 'name': '3004 豐達科'},
    {'file': 'get_trading_signal_3022.py', 'name': '3022 威強電'},
    {'file': 'get_trading_signal_3135.py', 'name': '3135 台股'},
    {'file': 'get_trading_signal_3138.py', 'name': '3138 耀登'},
    {'file': 'get_trading_signal_3260.py', 'name': '3260 威剛'},
    {'file': 'get_trading_signal_3491.py', 'name': '3491 台股'},
    {'file': 'get_trading_signal_4967.py', 'name': '4967 台股'},
    {'file': 'get_trading_signal_5371.py', 'name': '5371 台股'},
    {'file': 'get_trading_signal_6446.py', 'name': '6446 藥華藥'},
    {'file': 'get_trading_signal_6668.py', 'name': '6668 台股'},
    {'file': 'get_trading_signal_8222.py', 'name': '8222 台股'},
    {'file': 'get_trading_signal_2314.py', 'name': '2314 台揚'},
    {'file': 'get_trading_signal_3105.py', 'name': '3105 穩懋'},
    {'file': 'get_trading_signal_6271.py', 'name': '6271 同欣電'},
    {'file': 'get_trading_signal_4971.py', 'name': '4971 IET-KY'},
    {'file': 'get_trading_signal_1326.py', 'name': '1326 台塑化'},
    {'file': 'get_trading_signal_3533.py', 'name': '3533 嘉澤'},
    {'file': 'get_trading_signal_2059.py', 'name': '2059 川湖'},
    {'file': 'get_trading_signal_1514.py', 'name': '1514 亞力'},
    {'file': 'get_trading_signal_8046.py', 'name': '8046 南電'},
    {'file': 'get_trading_signal_6187.py', 'name': '6187 環球晶'},
    # {'file': 'get_trading_signal_7769.py', 'name': '7769 霖揚'},  # Insufficient data - disabled 2026-03-02
    # Added 2026-03-02 - XGBoost batch trained stocks
    {'file': 'get_trading_signal_3563.py', 'name': '3563 牧德'},
    {'file': 'get_trading_signal_3576.py', 'name': '3576 聯合再生'},
    {'file': 'get_trading_signal_3615.py', 'name': '3615 安可'},
    {'file': 'get_trading_signal_3665.py', 'name': '3665 貿聯-KY'},
    {'file': 'get_trading_signal_4564.py', 'name': '4564 元翎'},
    {'file': 'get_trading_signal_4577.py', 'name': '4577 達航科技'},
    {'file': 'get_trading_signal_4768.py', 'name': '4768 晶呈科技'},
    {'file': 'get_trading_signal_4989.py', 'name': '4989 榮科'},
    {'file': 'get_trading_signal_4991.py', 'name': '4991 環宇-KY'},
    {'file': 'get_trading_signal_6220.py', 'name': '6220 岳豐'},
    {'file': 'get_trading_signal_6230.py', 'name': '6230 尼得科超眾'},
    {'file': 'get_trading_signal_6526.py', 'name': '6526 達發'},
    {'file': 'get_trading_signal_6789.py', 'name': '6789 采鈺'},
    {'file': 'get_trading_signal_6830.py', 'name': '6830 汎銓'},
    {'file': 'get_trading_signal_6877.py', 'name': '6877 鏵友益'},
    {'file': 'get_trading_signal_8438.py', 'name': '8438 昶昕'},
    {'file': 'get_trading_signal_8927.py', 'name': '8927 北基'},
]

def run_signal_and_capture(script_file, stock_name, output_file):
    """运行单个交易信号生成器并捕获输出到文件"""
    separator = "\n" + "=" * 100 + "\n"
    header = f"运行: {stock_name}"

    # 写入分隔符和标题
    output_file.write(separator)
    output_file.write(header + "\n")
    output_file.write("=" * 100 + "\n")

    print(f"运行: {stock_name}", flush=True)

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
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
            output_file.write(result.stdout)
            output_file.flush()  # 立即写入磁盘
            return True
        else:
            error_msg = f"[X] 运行失败:\n{result.stderr}\n"
            output_file.write(error_msg)
            output_file.flush()
            return False

    except subprocess.TimeoutExpired:
        error_msg = f"[X] 超时 (180秒)\n"
        output_file.write(error_msg)
        output_file.flush()
        return False
    except Exception as e:
        error_msg = f"[X] 错误: {e}\n"
        output_file.write(error_msg)
        output_file.flush()
        return False

if __name__ == "__main__":
    # 生成输出文件名
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    output_filename = f'taiwan_signals_output_{timestamp}.txt'

    print("=" * 100)
    print("批量运行所有台股交易信号生成器 (输出到TXT文件)")
    print("=" * 100)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总共 {len(SIGNAL_SCRIPTS)} 个台股")
    print(f"输出文件: {output_filename}")
    print("=" * 100)

    success_count = 0
    failed_stocks = []

    # 打开输出文件
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        # 写入文件头
        output_file.write("=" * 100 + "\n")
        output_file.write("台股交易信号批量生成报告\n")
        output_file.write("=" * 100 + "\n")
        output_file.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_file.write(f"总共股票: {len(SIGNAL_SCRIPTS)}\n")
        output_file.write("=" * 100 + "\n\n")

        # 运行所有脚本
        for i, script in enumerate(SIGNAL_SCRIPTS, 1):
            print(f"\n进度: [{i}/{len(SIGNAL_SCRIPTS)}]", flush=True)

            success = run_signal_and_capture(script['file'], script['name'], output_file)

            if success:
                success_count += 1
                print(f"   ✅ 成功")
            else:
                failed_stocks.append(script['name'])
                print(f"   ❌ 失败")

        # 写入汇总
        summary = f"\n\n{'=' * 100}\n"
        summary += "批量运行完成!\n"
        summary += "=" * 100 + "\n"
        summary += f"成功运行: {success_count}/{len(SIGNAL_SCRIPTS)}\n"
        summary += f"失败数量: {len(failed_stocks)}\n"

        if failed_stocks:
            summary += "\n失败的股票:\n"
            for stock in failed_stocks:
                summary += f"   - {stock}\n"

        summary += "\n所有台股信号生成完成!\n"

        output_file.write(summary)

    # 控制台输出汇总
    print("\n" + "=" * 100)
    print("批量运行完成!")
    print("=" * 100)
    print(f"成功运行: {success_count}/{len(SIGNAL_SCRIPTS)}")
    print(f"失败数量: {len(failed_stocks)}")

    if failed_stocks:
        print(f"\n失败的股票:")
        for stock in failed_stocks:
            print(f"   - {stock}")

    print(f"\n✅ 输出文件已保存: {output_filename}")
    print("\n所有台股信号生成完成!")
