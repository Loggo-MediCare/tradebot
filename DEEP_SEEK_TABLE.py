# pip install yfinance pandas
import re
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

# ====== Config ======
SIGNALS_TXT = "taiwan_signals_output_202601151615.txt"
OUT_CSV = "tw_buy_rank_with_analyst_targets.csv"

# 权重配置（可调整）
W_MODEL = 0.25    # 模型准确度权重
W_SIGNAL = 0.30   # 信号强度权重
W_TECH = 0.20     # 技术指标权重
W_VOLUME = 0.15   # 量能权重
W_PATTERN = 0.10  # 形态分析权重（负向）

# ====== Utilities ======
def safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        x = str(x).replace("%", "").replace("NT$", "").replace("$", "").strip()
        return float(x)
    except Exception:
        return np.nan

def clip01(x):
    if pd.isna(x):
        return np.nan
    return float(np.clip(x, 0.0, 1.0))

def normalize_minmax(series, lo=None, hi=None):
    """Min-max scale to [0,1], ignore NaN."""
    s = series.astype(float)
    if lo is None:
        lo = np.nanmin(s.values)
    if hi is None:
        hi = np.nanmax(s.values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series([np.nan]*len(s), index=s.index)
    return (s - lo) / (hi - lo)

# ====== 1) Parse BUY tickers from signals text ======
def extract_buy_blocks(text: str):
    """
    解析完整的信号文件，提取每个股票的完整区块
    """
    # 使用进度标记作为分隔符
    progress_pattern = r"进度: \[(\d+)/\d+\]"
    # 找到所有进度标记的位置
    progress_matches = list(re.finditer(progress_pattern, text))
    
    blocks = []
    for i, match in enumerate(progress_matches):
        start_pos = match.end()
        # 下一个进度标记的位置或文件结束
        if i + 1 < len(progress_matches):
            end_pos = progress_matches[i + 1].start()
        else:
            end_pos = len(text)
        
        block_content = text[start_pos:end_pos].strip()
        if block_content:
            blocks.append({
                'progress': match.group(0),
                'content': block_content
            })
    
    return blocks

def parse_block_for_features(block: str):
    """
    从单个股票区块提取特征
    """
    # 1. 提取股票代码
    ticker = None
    m = re.search(r"股票:\s*(\d{4}\.TW)", block)
    if m:
        ticker = m.group(1)
    else:
        # 尝试其他格式
        m2 = re.search(r"运行:\s*(\d{4})", block)
        if m2:
            ticker = f"{m2.group(1)}.TW"
    
    if not ticker:
        return None
    
    # 2. 检查是否为买入信号
    signal_patterns = [
        r"信号:\s*买入\s*\(BUY\)",
        r"信号:\s*买入",
        r"Signal:\s*BUY",
        r"🟢\s*信号:\s*买入"
    ]
    is_buy = any(re.search(pattern, block, re.IGNORECASE) for pattern in signal_patterns)
    if not is_buy:
        return None
    
    # 3. 提取模型准确度
    accuracy = np.nan
    acc_patterns = [
        r"模型準確度:\s*[🟢🟡🔴]?\s*AI準確度:\s*([0-9\.]+)/100",
        r"模型准确度:\s*[🟢🟡🔴]?\s*AI准确度:\s*([0-9\.]+)/100",
        r"AI準確度:\s*([0-9\.]+)/100"
    ]
    for pattern in acc_patterns:
        m = re.search(pattern, block)
        if m:
            accuracy = safe_float(m.group(1)) / 100.0
            break
    
    # 4. 提取信号强度
    signal_strength = np.nan
    strength_patterns = [
        r"强度:\s*([0-9\.]+)",
        r"AI 模型强度:\s*([0-9\.]+)\s*/\s*1\.00",
        r"模型输出动作值:\s*([\+\-]?[0-9\.]+)"
    ]
    for pattern in strength_patterns:
        m = re.search(pattern, block)
        if m:
            val = safe_float(m.group(1))
            # 如果是动作值，转换为0-1范围（假设-1到1）
            if "动作值" in pattern and val is not np.nan:
                signal_strength = (val + 1) / 2  # 映射到0-1
            else:
                signal_strength = clip01(val)
            break
    
    # 5. 提取价格
    price = np.nan
    price_patterns = [
        r"当前价格:\s*NT\$([0-9\.,]+)",
        r"价格:\s*NT\$([0-9\.,]+)",
        r"当前价格:\s*\$([0-9\.,]+)"
    ]
    for pattern in price_patterns:
        m = re.search(pattern, block)
        if m:
            price = safe_float(m.group(1))
            break
    
    # 6. 提取技术指标
    # RSI
    rsi = np.nan
    m = re.search(r"RSI\s*\(14\):\s*([0-9\.]+)", block)
    if m:
        rsi_val = safe_float(m.group(1))
        # RSI得分：50-80为理想范围
        if rsi_val is not np.nan:
            if rsi_val < 30:
                rsi_score = 0.2  # 超卖
            elif rsi_val < 50:
                rsi_score = 0.5  # 偏弱
            elif rsi_val < 70:
                rsi_score = 0.9  # 理想
            elif rsi_val < 80:
                rsi_score = 0.7  # 偏强
            else:
                rsi_score = 0.3  # 超买
            rsi = rsi_score
    
    # MACD
    macd_score = np.nan
    macd_patterns = [
        r"MACD:\s*([\+\-]?[0-9\.]+)\s*MACD Signal:\s*[\+\-]?[0-9\.]+\s*\[金叉\]",
        r"MACD金叉"
    ]
    if any(re.search(pattern, block) for pattern in macd_patterns):
        macd_score = 0.8
    elif re.search(r"MACD死叉", block):
        macd_score = 0.3
    else:
        macd_score = 0.5
    
    # 均线排列
    ma_score = np.nan
    if re.search(r"均线多头排列", block):
        ma_score = 0.9
    elif re.search(r"均线空头排列", block):
        ma_score = 0.3
    else:
        ma_score = 0.5
    
    # 布林带位置
    bb_score = np.nan
    m = re.search(r"当前价格位置:\s*([0-9\.]+)%", block)
    if m:
        bb_pos = safe_float(m.group(1))
        if bb_pos is not np.nan:
            # 20%-80%为理想范围
            if bb_pos < 20:
                bb_score = 0.3  # 接近下轨，可能超卖
            elif bb_pos < 40:
                bb_score = 0.6  # 偏弱
            elif bb_pos < 60:
                bb_score = 0.8  # 中性
            elif bb_pos < 80:
                bb_score = 0.7  # 偏强
            else:
                bb_score = 0.4  # 接近上轨，可能超买
    
    # 计算综合技术指标得分
    tech_scores = [s for s in [rsi, macd_score, ma_score, bb_score] if not pd.isna(s)]
    tech_score = np.mean(tech_scores) if tech_scores else np.nan
    
    # 7. 提取成交量信息
    volume_ratio = np.nan
    volume_patterns = [
        r"量比:\s*([0-9\.]+)x",
        r"成交量:\s*[0-9,]+\s*20日平均量:\s*[0-9,]+\s*\[([^]]+)\]",
        r"放量"
    ]
    
    # 先尝试提取量比数值
    m = re.search(r"量比:\s*([0-9\.]+)x", block)
    if m:
        volume_ratio = safe_float(m.group(1))
    else:
        # 根据描述判断
        if re.search(r"放量", block):
            volume_ratio = 1.5  # 估计值
        elif re.search(r"缩量", block):
            volume_ratio = 0.7  # 估计值
        elif re.search(r"正常", block):
            volume_ratio = 1.0
    
    # 成交量得分：1.0x为中性，越高越好
    volume_score = np.nan
    if volume_ratio is not np.nan:
        if volume_ratio < 0.5:
            volume_score = 0.3
        elif volume_ratio < 0.8:
            volume_score = 0.5
        elif volume_ratio < 1.2:
            volume_score = 0.7
        elif volume_ratio < 2.0:
            volume_score = 0.9
        else:
            volume_score = 1.0
    
    # 8. 提取蜡烛图形态风险
    pattern_risk = 0.5  # 默认中性
    
    # 看跌形态
    bearish_patterns = ["吊人線", "墓碑十字", "空頭吞噬", "看跌吞噬", "乌云盖顶", "黄昏之星"]
    bearish_count = sum(1 for pattern in bearish_patterns if pattern in block)
    
    # 看涨形态
    bullish_patterns = ["槌子線", "蜻蜓十字", "上升缺口", "看漲吞噬", "晨星", "曙光初現"]
    bullish_count = sum(1 for pattern in bullish_patterns if pattern in block)
    
    # 形态风险评分：0(低风险)-1(高风险)
    pattern_risk = np.clip(0.5 + 0.1*bearish_count - 0.08*bullish_count, 0.0, 1.0)
    
    # 9. 提取建议买入比例
    buy_ratio = np.nan
    m = re.search(r"建议买入比例:\s*([0-9\.]+)%", block)
    if m:
        buy_ratio = safe_float(m.group(1)) / 100.0
    else:
        m2 = re.search(r"买入\s*(\d+)%", block)
        if m2:
            buy_ratio = safe_float(m2.group(1)) / 100.0
    
    # 10. 提取MA50趋势
    ma50_trend = 0.5
    if re.search(r"MA50趨勢向上", block):
        ma50_trend = 0.8
    elif re.search(r"MA50趨勢向下", block):
        ma50_trend = 0.2
    
    return {
        "ticker": ticker,
        "model_accuracy": accuracy,
        "signal_strength": signal_strength,
        "price": price,
        "tech_score": tech_score,
        "volume_score": volume_score,
        "pattern_risk": pattern_risk,
        "buy_ratio": buy_ratio,
        "ma50_trend": ma50_trend,
        "raw_content": block[:500]  # 保存部分原始内容用于调试
    }

def parse_all_buy(text: str) -> pd.DataFrame:
    """解析所有买入信号"""
    blocks = extract_buy_blocks(text)
    rows = []
    
    for block_info in blocks:
        content = block_info['content']
        feat = parse_block_for_features(content)
        if feat:
            rows.append(feat)
    
    df = pd.DataFrame(rows)
    
    # 去重，保留第一个出现的
    df = df.drop_duplicates(subset=["ticker"], keep='first')
    
    return df

# ====== 2) 从yfinance获取分析师目标价 ======
def yf_get_analyst_targets(ticker_tw: str) -> dict:
    """使用yfinance获取分析师目标价信息"""
    # 台湾股票在yfinance中的代码格式：1101.TW
    try:
        tk = yf.Ticker(ticker_tw)
        info = tk.info
        
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        target_mean = info.get("targetMeanPrice")
        target_high = info.get("targetHighPrice")
        target_low = info.get("targetLowPrice")
        analyst_count = info.get("numberOfAnalystOpinions")
        recommendation = info.get("recommendationKey", "hold").lower()
        recommendation_mean = info.get("recommendationMean")
        
        # 计算上涨潜力
        upside_pct = np.nan
        if price and target_mean and price > 0:
            upside_pct = (target_mean / price - 1.0) * 100
        
        # 推荐得分：1.0(强力买入)到0.0(强力卖出)
        rec_score_map = {
            "strong buy": 1.0,
            "buy": 0.8,
            "outperform": 0.7,
            "hold": 0.5,
            "underperform": 0.3,
            "sell": 0.2,
            "strong sell": 0.0
        }
        
        rec_score = rec_score_map.get(recommendation, 0.5)
        
        return {
            "price_yf": safe_float(price),
            "target_mean": safe_float(target_mean),
            "target_high": safe_float(target_high),
            "target_low": safe_float(target_low),
            "analyst_count": safe_float(analyst_count),
            "recommendation": recommendation,
            "recommendation_mean": safe_float(recommendation_mean),
            "recommendation_score": rec_score,
            "upside_pct": upside_pct
        }
    except Exception as e:
        print(f"Warning: Could not fetch data for {ticker_tw}: {e}")
        return {
            "price_yf": np.nan,
            "target_mean": np.nan,
            "target_high": np.nan,
            "target_low": np.nan,
            "analyst_count": np.nan,
            "recommendation": "N/A",
            "recommendation_mean": np.nan,
            "recommendation_score": np.nan,
            "upside_pct": np.nan
        }

# ====== 3) 综合评分 ======
def build_rank_table(df: pd.DataFrame) -> pd.DataFrame:
    """构建排名表"""
    # 确保所有分数在0-1范围内
    df["model_score"] = df["model_accuracy"].apply(clip01)
    df["signal_score"] = df["signal_strength"].apply(clip01)
    df["tech_score"] = df["tech_score"].apply(clip01)
    df["volume_score"] = df["volume_score"].apply(clip01)
    df["pattern_score"] = 1.0 - df["pattern_risk"].astype(float)  # 风险转换为分数
    df["ma50_score"] = df["ma50_trend"].apply(clip01)
    df["buy_ratio_score"] = df["buy_ratio"].apply(clip01)
    
    # 分析师分数
    df["analyst_score"] = df["recommendation_score"].apply(clip01)
    
    # 上涨潜力分数（0-1）
    def upside_score_func(upside):
        if pd.isna(upside):
            return np.nan
        # -20%到+50%映射到0-1
        return np.clip((upside - (-20)) / (50 - (-20)), 0.0, 1.0)
    
    df["upside_score"] = df["upside_pct"].apply(upside_score_func)
    
    # 分析师覆盖度分数
    def coverage_score_func(count):
        if pd.isna(count):
            return np.nan
        # 0-20位分析师映射到0-1
        return np.clip(count / 20.0, 0.0, 1.0)
    
    df["coverage_score"] = df["analyst_count"].apply(coverage_score_func)
    
    # 综合分析师分数（推荐得分40% + 上涨潜力40% + 覆盖度20%）
    analyst_components = ["analyst_score", "upside_score", "coverage_score"]
    df["analyst_composite"] = df[analyst_components].apply(
        lambda row: np.nanmean([row[c] for c in analyst_components if not pd.isna(row[c])]), 
        axis=1
    )
    
    # 计算最终分数
    # 第一部分：AI信号相关（55%）
    ai_components = {
        "model_score": W_MODEL,
        "signal_score": W_SIGNAL,
        "tech_score": W_TECH
    }
    
    # 第二部分：市场因素（35%）
    market_components = {
        "volume_score": W_VOLUME,
        "pattern_score": W_PATTERN,
        "ma50_score": 0.05,
        "buy_ratio_score": 0.05
    }
    
    # 第三部分：分析师共识（10%）
    analyst_weight = 0.10
    
    def calculate_weighted_score(row, components_dict):
        total_score = 0.0
        total_weight = 0.0
        
        for component, weight in components_dict.items():
            score = row.get(component)
            if pd.notna(score):
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else np.nan
    
    df["ai_score"] = df.apply(lambda row: calculate_weighted_score(row, ai_components), axis=1)
    df["market_score"] = df.apply(lambda row: calculate_weighted_score(row, market_components), axis=1)
    
    # 最终分数计算
    # 如果有分析师数据: AI分数 * 0.55 + 市场分数 * 0.35 + 分析师分数 * 0.10
    # 如果没有分析师数据: 重新分配权重给 AI 和 市场 (AI: 0.60, 市场: 0.40)
    final_scores = []
    for _, row in df.iterrows():
        ai = row["ai_score"] if pd.notna(row["ai_score"]) else 0.5
        market = row["market_score"] if pd.notna(row["market_score"]) else 0.5
        analyst = row["analyst_composite"]

        # 检查是否有有效的分析师数据
        has_analyst_data = pd.notna(analyst) and pd.notna(row.get("target_mean"))

        if has_analyst_data:
            # 有分析师数据：使用原始权重
            final_score = ai * 0.55 + market * 0.35 + analyst * 0.10
        else:
            # 没有分析师数据：重新分配权重给 AI 和 市场
            # 原本: AI 55%, 市场 35%, 分析师 10%
            # 重新分配: AI 60%, 市场 40% (把分析师的10%分给AI和市场)
            final_score = ai * 0.60 + market * 0.40

        final_scores.append(final_score)
    
    df["final_score"] = final_scores
    
    # 添加风险等级
    def risk_level(score):
        if score >= 0.8:
            return "低风险"
        elif score >= 0.6:
            return "中低风险"
        elif score >= 0.4:
            return "中等风险"
        elif score >= 0.2:
            return "中高风险"
        else:
            return "高风险"
    
    df["risk_level"] = df["final_score"].apply(risk_level)
    
    # 排序
    df = df.sort_values("final_score", ascending=False, na_position="last").reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    
    # 重新排列列顺序
    columns_order = [
        "rank", "ticker", "final_score", "risk_level",
        "ai_score", "market_score", "analyst_composite",
        "model_accuracy", "signal_strength", "tech_score",
        "volume_score", "pattern_risk", "ma50_trend", "buy_ratio",
        "price", "price_yf", "target_mean", "upside_pct",
        "analyst_count", "recommendation"
    ]
    
    # 只保留存在的列
    existing_columns = [col for col in columns_order if col in df.columns]
    other_columns = [col for col in df.columns if col not in existing_columns and col not in columns_order]
    
    df = df[existing_columns + other_columns]
    
    return df

# ====== 主程序 ======
def main():
    print("开始解析信号文件...")
    
    # 1. 读取文件
    try:
        with open(SIGNALS_TXT, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"错误: 文件未找到: {SIGNALS_TXT}")
        return
    
    # 2. 解析所有买入信号
    buy_df = parse_all_buy(text)
    
    if buy_df.empty:
        print("在信号文件中未找到买入信号。")
        return
    
    print(f"找到 {len(buy_df)} 个买入信号股票")
    
    # 3. 获取分析师数据
    print("获取分析师目标价数据...")
    analyst_data = []
    for ticker in buy_df["ticker"]:
        data = yf_get_analyst_targets(ticker)
        analyst_data.append(data)
    
    analyst_df = pd.DataFrame(analyst_data)
    
    # 4. 合并数据
    df = pd.concat([buy_df.reset_index(drop=True), analyst_df.reset_index(drop=True)], axis=1)
    
    # 5. 构建排名表
    print("计算综合评分...")
    ranked_df = build_rank_table(df)
    
    # 6. 保存到CSV
    ranked_df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ 排名结果已保存到: {OUT_CSV}")
    
    # 7. 显示前20名
    print("\n" + "="*80)
    print("前20名推荐股票:")
    print("="*80)
    
    display_cols = ["rank", "ticker", "final_score", "risk_level", 
                   "price", "target_mean", "upside_pct", 
                   "analyst_count", "recommendation", "model_accuracy"]
    
    # 格式化显示
    display_df = ranked_df.head(20).copy()
    
    # 格式化百分比
    if "model_accuracy" in display_df.columns:
        display_df["model_accuracy"] = display_df["model_accuracy"].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
        )
    
    if "upside_pct" in display_df.columns:
        display_df["upside_pct"] = display_df["upside_pct"].apply(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
        )
    
    if "final_score" in display_df.columns:
        display_df["final_score"] = display_df["final_score"].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
        )
    
    print(display_df[display_cols].to_string(index=False))
    
    # 8. 生成简要报告
    print("\n" + "="*80)
    print("简要分析报告:")
    print("="*80)
    
    top_5 = ranked_df.head(5)
    print(f"1. 最佳推荐: {top_5.iloc[0]['ticker']} (分数: {top_5.iloc[0]['final_score']:.3f})")
    print(f"2. 高风险高回报: {ranked_df[ranked_df['pattern_risk'] > 0.7].iloc[0]['ticker'] if len(ranked_df[ranked_df['pattern_risk'] > 0.7]) > 0 else 'N/A'}")
    print(f"3. 最保守选择: {ranked_df[ranked_df['pattern_risk'] < 0.3].iloc[0]['ticker'] if len(ranked_df[ranked_df['pattern_risk'] < 0.3]) > 0 else 'N/A'}")
    
    # 统计
    avg_score = ranked_df["final_score"].mean()
    avg_upside = ranked_df["upside_pct"].mean()
    print(f"\n统计信息:")
    print(f"- 平均综合分数: {avg_score:.3f}")
    print(f"- 平均上涨潜力: {avg_upside:+.1f}%" if pd.notna(avg_upside) else "- 平均上涨潜力: N/A")
    print(f"- 高风险股票数: {len(ranked_df[ranked_df['risk_level'] == '高风险'])}")
    print(f"- 低风险股票数: {len(ranked_df[ranked_df['risk_level'] == '低风险'])}")

if __name__ == "__main__":
    main()