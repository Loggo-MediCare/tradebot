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
import yfinance as yf

# 修复 Windows 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 强制输出刷新（避免缓冲导致终端只显示最后几个股票）
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUNBUFFERED'] = '1'

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

# 台股信号生成器

#Stock	Company	Accuracy	Model	Signal File	Added to run_all
#6683.TWO	雍智科技	59.4%	XGBoost	✅ Created	✅
#6223.TWO	旺矽	54.6%	RandomForest	✅ Created	✅
#3044.TW	健鼎	59.8%	RandomFores

SIGNAL_SCRIPTS = [
    {'file': 'get_trading_signal_2330.py', 'name': '2330 台積電'},
    {'file': 'get_trading_signal_2317.py', 'name': '2317 鴻海'},
    {'file': 'get_trading_signal_2408.py', 'name': '2408 南亞科'},
    {'file': 'get_trading_signal_2308.py', 'name': '2308 台達電'},
    {'file': 'get_trading_signal_2313.py', 'name': '2313 華通'},
    {'file': 'get_trading_signal_2454.py', 'name': '2454 聯發科'},
    {'file': 'get_trading_signal_2337.py', 'name': '2337 旺宏'},
    {'file': 'get_trading_signal_2344.py', 'name': '2344 華邦電'},
    {'file': 'get_trading_signal_2367.py', 'name': '2367 燿華'},
    {'file': 'get_trading_signal_3481.py', 'name': '3481 群創'},
    {'file': 'get_trading_signal_2603.py', 'name': '2603 長榮'},
    {'file': 'get_trading_signal_6770.py', 'name': '6770 力積電'},
    {'file': 'get_trading_signal_3665.py', 'name': '3665 貿聯-KY'},
    {'file': 'get_trading_signal_3017.py', 'name': '3017 奇鋐'},
    {'file': 'get_trading_signal_3711.py', 'name': '3711 日月光投控'},
    {'file': 'get_trading_signal_3037.py', 'name': '3037 欣興'},
    {'file': 'get_trading_signal_2327.py', 'name': '2327 國巨'},
    {'file': 'get_trading_signal_2382.py', 'name': '2382 廣達'},
    {'file': 'get_trading_signal_3443.py', 'name': '3443 創意'},
    {'file': 'get_trading_signal_2383.py', 'name': '2383 台光電'},
    {'file': 'get_trading_signal_6442.py', 'name': '6442 光聖'},
    {'file': 'get_trading_signal_3661.py', 'name': '3661 世芯-KY'},
    {'file': 'get_trading_signal_6669.py', 'name': '6669 緯穎'},
    {'file': 'get_trading_signal_3231.py', 'name': '3231 緯創'},
    {'file': 'get_trading_signal_2303.py', 'name': '2303 聯電'},
    {'file': 'get_trading_signal_2368.py', 'name': '2368 金像電'},
    {'file': 'get_trading_signal_2345.py', 'name': '2345 智邦'},
    {'file': 'get_trading_signal_1303.py', 'name': '1303 南亞'},
    {'file': 'get_trading_signal_2360.py', 'name': '2360 致茂'},
    {'file': 'get_trading_signal_2449.py', 'name': '2449 京元電子'},
    {'file': 'get_trading_signal_6443.py', 'name': '6443 元晶'},
    {'file': 'get_trading_signal_4989.py', 'name': '4989 榮科'},
    {'file': 'get_trading_signal_6285.py', 'name': '6285 啟碁'},
    {'file': 'get_trading_signal_3715.py', 'name': '3715 定穎投控'},
    {'file': 'get_trading_signal_3563.py', 'name': '3563 牧德'},
    {'file': 'get_trading_signal_3653.py', 'name': '3653 健策'},
    {'file': 'get_trading_signal_2891.py', 'name': '2891 中信金'},
    {'file': 'get_trading_signal_6239.py', 'name': '6239 力成'},
    {'file': 'get_trading_signal_3533.py', 'name': '3533 嘉澤'},
    {'file': 'get_trading_signal_8069.py', 'name': '8069 元太'},
    {'file': 'get_trading_signal_6683.py', 'name': '6683 雍智科技'},
    {'file': 'get_trading_signal_6223.py', 'name': '6223 旺矽'},
    {'file': 'get_trading_signal_3363.py', 'name': '3363 上詮'},
    {'file': 'get_trading_signal_3449.py', 'name': '3449 鈺德'},
    {'file': 'get_trading_signal_5483.py', 'name': '5483 中美晶'},
    {'file': 'get_trading_signal_6163.py', 'name': '6163 華電網'},
    {'file': 'get_trading_signal_7709.py', 'name': '7709 榮田'},
    {'file': 'get_trading_signal_7717.py', 'name': '7717 萊德光電'},
    {'file': 'get_trading_signal_3260.py', 'name': '3260 威剛'},
    {'file': 'get_trading_signal_3491.py', 'name': '3491 昇達科'},
    {'file': 'get_trading_signal_5371.py', 'name': '5371 中光電'},
    {'file': 'get_trading_signal_3105.py', 'name': '3105 穩懋'},
    {'file': 'get_trading_signal_4971.py', 'name': '4971 IET-KY'},
    {'file': 'get_trading_signal_6187.py', 'name': '6187 環球晶'},
    {'file': 'get_trading_signal_3615.py', 'name': '3615 安可'},
    {'file': 'get_trading_signal_4577.py', 'name': '4577 達航科技'},
    {'file': 'get_trading_signal_4768.py', 'name': '4768 晶呈科技'},
    {'file': 'get_trading_signal_4991.py', 'name': '4991 環宇-KY'},
    {'file': 'get_trading_signal_6220.py', 'name': '6220 岳豐'},
    {'file': 'get_trading_signal_6877.py', 'name': '6877 鏵友益'},
    {'file': 'get_trading_signal_8927.py', 'name': '8927 北基'},
    {'file': 'get_trading_signal_1519.py', 'name': '1519 華城'},
    {'file': 'get_trading_signal_6805.py', 'name': '6805 富世達'},
    {'file': 'get_trading_signal_6789.py', 'name': '6789 采鈺'},
    {'file': 'get_trading_signal_8021.py', 'name': '8021 尖點'},
    {'file': 'get_trading_signal_3006.py', 'name': '3006 晶豪科'},
    {'file': 'get_trading_signal_6830.py', 'name': '6830 汎銓'},
    {'file': 'get_trading_signal_2357.py', 'name': '2357 華碩'},
    {'file': 'get_trading_signal_3030.py', 'name': '3030 德律'},
    {'file': 'get_trading_signal_2409.py', 'name': '2409 友達'},
    {'file': 'get_trading_signal_2376.py', 'name': '2376 技嘉'},
    {'file': 'get_trading_signal_8210.py', 'name': '8210 勤誠'},
    {'file': 'get_trading_signal_6446.py', 'name': '6446 藥華藥'},
    {'file': 'get_trading_signal_1326.py', 'name': '1326 台塑化'},
    {'file': 'get_trading_signal_8046.py', 'name': '8046 南電'},
    {'file': 'get_trading_signal_1605.py', 'name': '1605 華新'},
    {'file': 'get_trading_signal_1301.py', 'name': '1301 台塑'},
    {'file': 'get_trading_signal_2059.py', 'name': '2059 川湖'},
    {'file': 'get_trading_signal_6781.py', 'name': '6781 AES-KY'},
    {'file': 'get_trading_signal_2884.py', 'name': '2884 玉山金'},
    {'file': 'get_trading_signal_6271.py', 'name': '6271 同欣電'},
    {'file': 'get_trading_signal_6515.py', 'name': '6515 穎崴'},
    {'file': 'get_trading_signal_2002.py', 'name': '2002 中鋼'},
    {'file': 'get_trading_signal_6526.py', 'name': '6526 達發'},
    {'file': 'get_trading_signal_3138.py', 'name': '3138 耀登'},
    {'file': 'get_trading_signal_8150.py', 'name': '8150 南茂'},
    {'file': 'get_trading_signal_1101.py', 'name': '1101 台泥'},
    {'file': 'get_trading_signal_2890.py', 'name': '2890 永豐金'},
    {'file': 'get_trading_signal_3044.py', 'name': '3044 健鼎'},
    {'file': 'get_trading_signal_4967.py', 'name': '4967 十銓'},
    {'file': 'get_trading_signal_2451.py', 'name': '2451 創見資訊'},
    {'file': 'get_trading_signal_8110.py', 'name': '8110 華東'},
    {'file': 'get_trading_signal_2385.py', 'name': '2385 群光'},
    {'file': 'get_trading_signal_4938.py', 'name': '4938 和碩'},
    {'file': 'get_trading_signal_3576.py', 'name': '3576 聯合再生'},
    {'file': 'get_trading_signal_2634.py', 'name': '2634 漢翔'},
    {'file': 'get_trading_signal_1514.py', 'name': '1514 亞力'},
    {'file': 'get_trading_signal_4722.py', 'name': '4722 国精化'},
    {'file': 'get_trading_signal_6472.py', 'name': '6472 保瑞'},
    {'file': 'get_trading_signal_8131.py', 'name': '8131 福懋科'},
    {'file': 'get_trading_signal_6230.py', 'name': '6230 尼得科超眾'},
    {'file': 'get_trading_signal_2363.py', 'name': '2363 矽統'},
    {'file': 'get_trading_signal_6209.py', 'name': '6209 今國光'},
    {'file': 'get_trading_signal_3135.py', 'name': '3135 凌航'},
    {'file': 'get_trading_signal_6269.py', 'name': '6269 台郡'},
    {'file': 'get_trading_signal_8438.py', 'name': '8438 昶昕'},
    {'file': 'get_trading_signal_4564.py', 'name': '4564 元翎'},
    {'file': 'get_trading_signal_4540.py', 'name': '4540 全球傳動'},
    {'file': 'get_trading_signal_8499.py', 'name': '8499 鼎炫-KY'},
    {'file': 'get_trading_signal_6477.py', 'name': '6477 安集'},
    {'file': 'get_trading_signal_3004.py', 'name': '3004 豐達科'},
    {'file': 'get_trading_signal_4746.py', 'name': '4746 台耀'},
    {'file': 'get_trading_signal_8222.py', 'name': '8222 寶一'},
    {'file': 'get_trading_signal_3022.py', 'name': '3022 威強電'},
    {'file': 'get_trading_signal_6668.py', 'name': '6668 中揚光'},
    {'file': 'get_trading_signal_2314.py', 'name': '2314 台揚'},

    # Gas/Petrochemical Stocks
    {'file': 'get_trading_signal_1314.py', 'name': '1314 中石化'},
    {'file': 'get_trading_signal_8908.py', 'name': '8908 欣雄'},
    {'file': 'get_trading_signal_9931.py', 'name': '9931 欣高'},
    {'file': 'get_trading_signal_8917.py', 'name': '8917 欣泰'},
    {'file': 'get_trading_signal_6505.py', 'name': '6505 台塑化'},
    {'file': 'get_trading_signal_9918.py', 'name': '9918 欣天然'},

    # Additional trained stocks (added 2026-05-27)
    {'file': 'get_trading_signal_1425.py', 'name': '1425 中福'},
    {'file': 'get_trading_signal_1471.py', 'name': '1471 首利'},
    {'file': 'get_trading_signal_1711.py', 'name': '1711 永光'},
    {'file': 'get_trading_signal_1727.py', 'name': '1727 中華化'},
    {'file': 'get_trading_signal_2301.py', 'name': '2301 光寶科'},
    {'file': 'get_trading_signal_2369.py', 'name': '2369 菱生'},
    {'file': 'get_trading_signal_2399.py', 'name': '2399 映泰'},
    {'file': 'get_trading_signal_2426.py', 'name': '2426 鼎元'},
    {'file': 'get_trading_signal_2467.py', 'name': '2467 志聖'},
    {'file': 'get_trading_signal_2485.py', 'name': '2485 兆赫'},
    {'file': 'get_trading_signal_2489.py', 'name': '2489 瑞軒'},
    {'file': 'get_trading_signal_2645.py', 'name': '2645 漢翔'},
    {'file': 'get_trading_signal_2886.py', 'name': '2886 兆豐金'},
    {'file': 'get_trading_signal_3081.py', 'name': '3081 聯亞科技'},
    {'file': 'get_trading_signal_3092.py', 'name': '3092 鴻碩'},
    {'file': 'get_trading_signal_3163.py', 'name': '3163 波若威'},
    {'file': 'get_trading_signal_3450.py', 'name': '3450 聯鈞'},
    {'file': 'get_trading_signal_3535.py', 'name': '3535 晶彩科'},
    {'file': 'get_trading_signal_4979.py', 'name': '4979 華星光通'},
    {'file': 'get_trading_signal_5269.py', 'name': '5269 祥碩'},
    {'file': 'get_trading_signal_6108.py', 'name': '6108 競國'},
    {'file': 'get_trading_signal_6133.py', 'name': '6133 金橋'},
    {'file': 'get_trading_signal_6139.py', 'name': '6139 亞翔'},
    {'file': 'get_trading_signal_6205.py', 'name': '6205 詮欣'},
    {'file': 'get_trading_signal_6213.py', 'name': '6213 聯茂'},
    {'file': 'get_trading_signal_6274.py', 'name': '6274 台燿'},
    {'file': 'get_trading_signal_6282.py', 'name': '6282 康舒'},
    {'file': 'get_trading_signal_6456.py', 'name': '6456 GIS-KY'},
    {'file': 'get_trading_signal_6831.py', 'name': '6831 騰雲'},
    {'file': 'get_trading_signal_6994.py', 'name': '6994 環科'},
    {'file': 'get_trading_signal_7610.py', 'name': '7610 聯友金屬'},
    {'file': 'get_trading_signal_7744.py', 'name': '7744 致伸'},
    {'file': 'get_trading_signal_7769.py', 'name': '7769 霖揚'},
    {'file': 'get_trading_signal_8033.py', 'name': '8033 雷虎'},
    {'file': 'get_trading_signal_8039.py', 'name': '8039 台虹'},
    {'file': 'get_trading_signal_8064.py', 'name': '8064 東捷科技'},
    {'file': 'get_trading_signal_8103.py', 'name': '8103 瀚荃'},
    {'file': 'get_trading_signal_8112.py', 'name': '8112 至上'},

    # New signal scripts (added 2026-05-27) - PPO models
    {'file': 'get_trading_signal_1815.py', 'name': '1815 富邦媒'},
    {'file': 'get_trading_signal_3293.py', 'name': '3293 鈊象'},
    {'file': 'get_trading_signal_3360.py', 'name': '3360 尚志'},
    {'file': 'get_trading_signal_3374.py', 'name': '3374 精材'},
    {'file': 'get_trading_signal_3630.py', 'name': '3630 新巨'},
    {'file': 'get_trading_signal_3690.py', 'name': '3690 美律'},
    {'file': 'get_trading_signal_3706.py', 'name': '3706 神達'},
    {'file': 'get_trading_signal_3707.py', 'name': '3707 漢磊'},
    {'file': 'get_trading_signal_4167.py', 'name': '4167 松瑞藥'},
    {'file': 'get_trading_signal_4541.py', 'name': '4541 全球傳動'},
    {'file': 'get_trading_signal_4973.py', 'name': '4973 廣宇'},
    {'file': 'get_trading_signal_6104.py', 'name': '6104 創源'},
    {'file': 'get_trading_signal_6265.py', 'name': '6265 方土昇'},
    {'file': 'get_trading_signal_6603.py', 'name': '6603 富奇想'},
    {'file': 'get_trading_signal_6829.py', 'name': '6829 勝麗'},
    {'file': 'get_trading_signal_6949.py', 'name': '6949 連展投控'},
    {'file': 'get_trading_signal_7728.py', 'name': '7728 中磊'},
    {'file': 'get_trading_signal_8038.py', 'name': '8038 長園科'},
    {'file': 'get_trading_signal_8042.py', 'name': '8042 金山電'},
    {'file': 'get_trading_signal_8074.py', 'name': '8074 特技'},
    {'file': 'get_trading_signal_8271.py', 'name': '8271 宇瞻'},
    {'file': 'get_trading_signal_8292.py', 'name': '8292 邦特'},
    {'file': 'get_trading_signal_8358.py', 'name': '8358 金居'},
    {'file': 'get_trading_signal_8377.py', 'name': '8377 建漢'},
    {'file': 'get_trading_signal_8431.py', 'name': '8431 匯僑'},
    {'file': 'get_trading_signal_8450.py', 'name': '8450 霸機科'},

    # New signal scripts (added 2026-05-27) - XGBoost models
    {'file': 'get_trading_signal_2380.py', 'name': '2380 虹光'},
    {'file': 'get_trading_signal_6135.py', 'name': '6135 新麗'},
    {'file': 'get_trading_signal_9136.py', 'name': '9136 巨騰-DR'},

    # ── Batch 2 新訓練股票 (added 2026-05-28) — PPO ────────────────────────
    {'file': 'get_trading_signal_4760.py', 'name': '4760 勤凱'},
    {'file': 'get_trading_signal_8291.py', 'name': '8291 尚茂'},
    {'file': 'get_trading_signal_3042.py', 'name': '3042 晶技'},
    {'file': 'get_trading_signal_6284.py', 'name': '6284 佳邦'},
    {'file': 'get_trading_signal_3485.py', 'name': '3485 敘豐'},
    {'file': 'get_trading_signal_6570.py', 'name': '6570 維田'},
    {'file': 'get_trading_signal_6207.py', 'name': '6207 齊科'},
    {'file': 'get_trading_signal_3026.py', 'name': '3026 禾伸堂'},
    {'file': 'get_trading_signal_6196.py', 'name': '6196 帆宣'},
    {'file': 'get_trading_signal_4927.py', 'name': '4927 泰鼎-KY'},
    {'file': 'get_trading_signal_6173.py', 'name': '6173 信昌電'},
    {'file': 'get_trading_signal_2464.py', 'name': '2464 盟立'},
    {'file': 'get_trading_signal_3236.py', 'name': '3236 千如'},
    {'file': 'get_trading_signal_6658.py', 'name': '6658 聯策'},
    {'file': 'get_trading_signal_6727.py', 'name': '6727 亞泰金屬'},

    # ── Batch 3 新訓練股票 (added 2026-05-28) — PPO ────────────────────────
    {'file': 'get_trading_signal_2492.py', 'name': '2492 華新科'},
    {'file': 'get_trading_signal_2472.py', 'name': '2472 立隆電'},
    {'file': 'get_trading_signal_2375.py', 'name': '2375 凱美'},
    {'file': 'get_trading_signal_6449.py', 'name': '6449 鉅邦'},
    {'file': 'get_trading_signal_6175.py', 'name': '6175 立敦'},
    {'file': 'get_trading_signal_6127.py', 'name': '6127 九豪'},
    {'file': 'get_trading_signal_3357.py', 'name': '3357 臺慶科'},
    {'file': 'get_trading_signal_2478.py', 'name': '2478 大毅'},
    {'file': 'get_trading_signal_3090.py', 'name': '3090 日電貿'},
    {'file': 'get_trading_signal_6224.py', 'name': '6224 聚鼎'},
    {'file': 'get_trading_signal_8043.py', 'name': '8043 蜜望寶'},
]

def run_signal(script_file, stock_name, peg_ratio=None):
    """运行单个交易信号生成器"""
    print("\n" + "=" * 100, flush=True)
    print(f"运行: {stock_name}", flush=True)
    if peg_ratio is not None and peg_ratio > 0:
        print(f"PEG 比率: {peg_ratio:.2f}", flush=True)
    else:
        print(f"PEG 比率: N/A", flush=True)
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
    peg_ratios = {}  # 存储 PEG 比率

    # TWO 交易所股票清單（其餘使用 TW）- 根據實際 signal scripts 的 TICKER 設定
    TWO_STOCKS = {'3498', '3615', '4533', '4577', '4768', '4908', '4991', '5011',
                  '6134', '6187', '6220', '6530', '6877', '7805', '8086', '8908', '8917', '8927',
                  '3163', '4979', '7744', '8064'}

    for i, script in enumerate(SIGNAL_SCRIPTS, 1):
        print(f"\n进度: [{i}/{len(SIGNAL_SCRIPTS)}]", flush=True)

        # 提取股票代碼 (取名稱第一個單詞)
        stock_number = script['name'].split()[0]

        # 根據股票編號判斷交易所後綴
        if stock_number in TWO_STOCKS:
            ticker_symbol = f"{stock_number}.TWO"
        else:
            ticker_symbol = f"{stock_number}.TW"

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
    print("\n" + "=" * 100, flush=True)
    print("批量运行完成!", flush=True)
    print("=" * 100, flush=True)
    print(f"成功运行: {success_count}/{len(SIGNAL_SCRIPTS)}", flush=True)
    print(f"失败数量: {len(failed_stocks)}", flush=True)

    if failed_stocks:
        print(f"\n失败的股票:", flush=True)
        for stock in failed_stocks:
            print(f"   - {stock}", flush=True)

    # 顯示 PEG 比率總結
    print("\n" + "=" * 100, flush=True)
    print("PEG 比率總結 (Price/Earnings to Growth Ratio)", flush=True)
    print("=" * 100, flush=True)

    # 過濾出有效的 PEG 比率並排序
    valid_pegs = [(name, peg) for name, peg in peg_ratios.items() if peg is not None and peg > 0]
    valid_pegs.sort(key=lambda x: x[1])  # 按 PEG 比率排序

    if valid_pegs:
        print(f"\n{'股票':<40} {'PEG 比率':>10}", flush=True)
        print("-" * 52, flush=True)
        for name, peg in valid_pegs:
            peg_status = "低估" if peg < 1.0 else "合理" if peg < 2.0 else "高估"
            print(f"{name:<40} {peg:>10.2f}  ({peg_status})", flush=True)

        avg_peg = sum(peg for _, peg in valid_pegs) / len(valid_pegs)
        print("-" * 52, flush=True)
        print(f"{'平均 PEG 比率:':<40} {avg_peg:>10.2f}", flush=True)
        print(f"\nPEG < 1.0: 可能被低估", flush=True)
        print(f"PEG 1.0-2.0: 估值合理", flush=True)
        print(f"PEG > 2.0: 可能被高估", flush=True)

    na_count = len([p for p in peg_ratios.values() if p is None or p <= 0])
    if na_count > 0:
        print(f"\n{na_count} 支股票無 PEG 比率數據", flush=True)

    print("\n所有台股信号生成完成!", flush=True)
