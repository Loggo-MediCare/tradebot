"""
测试所有股票的 FinBERT 情绪分析功能
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from finbert_sentiment import FinBERTAnalyzer, SEARCH_MAPPING

def test_finbert_setup():
    """测试 FinBERT 是否正确安装和配置"""
    print("=" * 70)
    print("FinBERT 配置测试")
    print("=" * 70)

    # 1. 测试 FinBERT 初始化
    print("\n1. 初始化 FinBERT 模型...")
    try:
        analyzer = FinBERTAnalyzer()
        if analyzer.nlp_finbert is not None:
            print("   ✅ FinBERT 模型加载成功!")
        else:
            print("   ❌ FinBERT 模型加载失败")
            return False
    except Exception as e:
        print(f"   ❌ FinBERT 初始化失败: {e}")
        return False

    # 2. 测试股票映射
    print("\n2. 检查股票映射表...")
    tw_stocks = [k for k in SEARCH_MAPPING.keys() if k.endswith('.TW')]
    us_stocks = [k for k in SEARCH_MAPPING.keys() if not k.endswith('.TW') and not k.endswith('.DE')]

    print(f"   ✅ 台股映射: {len(tw_stocks)} 支")
    print(f"   ✅ 美股映射: {len(us_stocks)} 支")
    print(f"   ✅ 总计: {len(SEARCH_MAPPING)} 支股票")

    # 3. 显示所有台股映射
    print("\n3. 台股 FinBERT 映射列表:")
    for stock in sorted(tw_stocks):
        search_ticker, search_name = SEARCH_MAPPING[stock]
        display_name = search_name if search_name else search_ticker
        print(f"   {stock:12} -> {display_name}")

    # 4. 测试情绪分析 (使用简单文本)
    print("\n4. 测试情绪分析功能...")
    test_texts = [
        "The company reported strong earnings and revenue growth.",
        "Stock prices fell amid market uncertainty and weak guidance.",
        "The company announced a new product launch."
    ]

    for text in test_texts:
        try:
            result = analyzer.nlp_finbert(text[:512])[0]
            label = result['label']
            score = result['score']
            print(f"   ✅ '{text[:50]}...' -> {label} ({score:.2f})")
        except Exception as e:
            print(f"   ❌ 分析失败: {e}")
            return False

    print("\n" + "=" * 70)
    print("✅ FinBERT 配置测试通过!")
    print("=" * 70)
    return True

def list_stocks_with_finbert():
    """列出所有支持 FinBERT 的股票"""
    print("\n" + "=" * 70)
    print("所有支持 FinBERT 的股票")
    print("=" * 70)

    # 按类别分组
    tw_stocks = sorted([k for k in SEARCH_MAPPING.keys() if k.endswith('.TW')])
    us_stocks = sorted([k for k in SEARCH_MAPPING.keys() if not k.endswith('.TW') and not k.endswith('.DE')])
    eu_stocks = sorted([k for k in SEARCH_MAPPING.keys() if k.endswith('.DE')])

    print(f"\n📊 台股 ({len(tw_stocks)} 支):")
    for i, stock in enumerate(tw_stocks, 1):
        search_ticker, search_name = SEARCH_MAPPING[stock]
        display_name = search_name if search_name else search_ticker
        print(f"   {i:2d}. {stock:12} -> {display_name}")

    print(f"\n🇺🇸 美股 ({len(us_stocks)} 支):")
    for i, stock in enumerate(us_stocks, 1):
        print(f"   {i:2d}. {stock}")

    if eu_stocks:
        print(f"\n🇪🇺 欧股 ({len(eu_stocks)} 支):")
        for i, stock in enumerate(eu_stocks, 1):
            search_ticker, search_name = SEARCH_MAPPING[stock]
            display_name = search_name if search_name else search_ticker
            print(f"   {i:2d}. {stock:12} -> {display_name}")

    print(f"\n📈 总计: {len(SEARCH_MAPPING)} 支股票支持 FinBERT 情绪分析")

if __name__ == '__main__':
    print("\n🚀 FinBERT 完整测试\n")

    # 运行测试
    success = test_finbert_setup()

    if success:
        # 列出所有股票
        list_stocks_with_finbert()

        print("\n" + "=" * 70)
        print("🎉 所有测试完成!")
        print("=" * 70)
        print("\n💡 提示:")
        print("   • FinBERT 已正确配置并可用于所有股票")
        print("   • 运行任何 get_trading_signal_*.py 即可使用情绪分析")
        print("   • 情绪分析会自动从 Google News 抓取相关新闻")
    else:
        print("\n" + "=" * 70)
        print("❌ 测试失败 - FinBERT 配置有问题")
        print("=" * 70)
        print("\n💡 解决方案:")
        print("   pip install transformers torch")
