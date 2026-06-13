"""
模型準確度追蹤器
用於記錄和顯示AI模型的歷史準確度
"""

import json
import os
from datetime import datetime
from pathlib import Path


class ModelAccuracyTracker:
    """追蹤和管理模型準確度數據"""

    def __init__(self, symbol, model_type="PPO"):
        """
        初始化追蹤器

        Parameters:
        -----------
        symbol : str
            股票代號 (e.g., "2330.TW", "HTGC")
        model_type : str
            模型類型 (預設: "PPO")
        """
        self.symbol = symbol
        self.model_type = model_type
        self.accuracy_file = Path(__file__).parent / f"model_accuracy_{symbol.replace('.', '_')}.json"

    def load_accuracy_data(self):
        """加載準確度數據"""
        if self.accuracy_file.exists():
            try:
                with open(self.accuracy_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  無法讀取準確度數據: {e}")
                return self._get_default_data()
        else:
            return self._get_default_data()

    def _get_default_data(self):
        """返回預設數據結構"""
        return {
            'symbol': self.symbol,
            'model_type': self.model_type,
            'training_accuracy': None,
            'validation_accuracy': None,
            'backtest_accuracy': None,
            'win_rate': None,
            'sharpe_ratio': None,
            'total_signals': 0,
            'correct_signals': 0,
            'live_accuracy': None,
            'last_updated': None,
            'history': []
        }

    def save_accuracy_data(self, data):
        """保存準確度數據"""
        try:
            with open(self.accuracy_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"❌ 無法保存準確度數據: {e}")
            return False

    def update_training_stats(self, training_acc=None, validation_acc=None, backtest_acc=None,
                             win_rate=None, sharpe_ratio=None):
        """
        更新訓練統計數據

        Parameters:
        -----------
        training_acc : float
            訓練準確度 (0-100)
        validation_acc : float
            驗證準確度 (0-100)
        backtest_acc : float
            回測準確度 (0-100)
        win_rate : float
            勝率 (0-100)
        sharpe_ratio : float
            夏普比率
        """
        data = self.load_accuracy_data()

        if training_acc is not None:
            data['training_accuracy'] = training_acc
        if validation_acc is not None:
            data['validation_accuracy'] = validation_acc
        if backtest_acc is not None:
            data['backtest_accuracy'] = backtest_acc
        if win_rate is not None:
            data['win_rate'] = win_rate
        if sharpe_ratio is not None:
            data['sharpe_ratio'] = sharpe_ratio

        data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.save_accuracy_data(data)

    def record_prediction(self, predicted_action, actual_result=None):
        """
        記錄一次預測

        Parameters:
        -----------
        predicted_action : str
            預測動作 ("BUY", "SELL", "HOLD")
        actual_result : bool, optional
            實際結果是否正確
        """
        data = self.load_accuracy_data()
        data['total_signals'] += 1

        if actual_result is not None:
            if actual_result:
                data['correct_signals'] += 1

            # 更新實時準確度
            if data['total_signals'] > 0:
                data['live_accuracy'] = (data['correct_signals'] / data['total_signals']) * 100

        # 記錄歷史
        data['history'].append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'predicted_action': predicted_action,
            'result': actual_result
        })

        # 只保留最近100筆記錄
        if len(data['history']) > 100:
            data['history'] = data['history'][-100:]

        data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.save_accuracy_data(data)

    def get_accuracy_summary(self):
        """
        獲取準確度摘要

        Returns:
        --------
        dict: 包含各種準確度指標
        """
        data = self.load_accuracy_data()

        summary = {
            'symbol': self.symbol,
            'model_type': self.model_type,
            'training_accuracy': data.get('training_accuracy'),
            'validation_accuracy': data.get('validation_accuracy'),
            'backtest_accuracy': data.get('backtest_accuracy'),
            'live_accuracy': data.get('live_accuracy'),
            'win_rate': data.get('win_rate'),
            'sharpe_ratio': data.get('sharpe_ratio'),
            'total_signals': data.get('total_signals', 0),
            'correct_signals': data.get('correct_signals', 0),
            'last_updated': data.get('last_updated')
        }

        # 計算綜合評分
        summary['overall_score'] = self._calculate_overall_score(summary)

        return summary

    def _calculate_overall_score(self, summary):
        """計算綜合評分 (0-100)"""
        scores = []
        weights = []

        if summary['backtest_accuracy'] is not None:
            scores.append(summary['backtest_accuracy'])
            weights.append(0.4)  # 回測準確度權重40%

        if summary['live_accuracy'] is not None:
            scores.append(summary['live_accuracy'])
            weights.append(0.3)  # 實時準確度權重30%

        if summary['win_rate'] is not None:
            scores.append(summary['win_rate'])
            weights.append(0.2)  # 勝率權重20%

        if summary['sharpe_ratio'] is not None:
            # 將夏普比率轉換為0-100分 (假設3.0為滿分)
            sharpe_score = min(summary['sharpe_ratio'] / 3.0 * 100, 100)
            scores.append(sharpe_score)
            weights.append(0.1)  # 夏普比率權重10%

        if not scores:
            return None

        # 正規化權重
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # 加權平均
        overall = sum(s * w for s, w in zip(scores, weights))

        return round(overall, 2)

    def format_accuracy_display(self):
        """
        格式化準確度顯示

        Returns:
        --------
        str: 格式化的輸出文本
        """
        summary = self.get_accuracy_summary()

        output = []
        output.append("=" * 80)
        output.append("🎯 AI 模型準確度統計")
        output.append("=" * 80)

        # 綜合評分
        if summary['overall_score'] is not None:
            score = summary['overall_score']
            if score >= 80:
                rating = "🌟🌟🌟🌟🌟 優秀"
                color = "🟢"
            elif score >= 70:
                rating = "🌟🌟🌟🌟 良好"
                color = "🟢"
            elif score >= 60:
                rating = "🌟🌟🌟 中等"
                color = "🟡"
            elif score >= 50:
                rating = "🌟🌟 普通"
                color = "🟡"
            else:
                rating = "🌟 待改進"
                color = "🔴"

            output.append(f"{color} 綜合評分:      {score:.1f}/100  {rating}")
        else:
            output.append("⚪ 綜合評分:      尚無數據")

        output.append("")

        # 各項指標
        if summary['backtest_accuracy'] is not None:
            output.append(f"📊 回測準確度:    {summary['backtest_accuracy']:.2f}%")
        else:
            output.append(f"📊 回測準確度:    尚未測試")

        if summary['live_accuracy'] is not None:
            output.append(f"🎯 實時準確度:    {summary['live_accuracy']:.2f}%  (共{summary['total_signals']}次預測)")
        else:
            output.append(f"🎯 實時準確度:    尚無數據")

        if summary['win_rate'] is not None:
            output.append(f"💰 勝率:          {summary['win_rate']:.2f}%")
        else:
            output.append(f"💰 勝率:          尚未統計")

        if summary['sharpe_ratio'] is not None:
            output.append(f"📈 夏普比率:      {summary['sharpe_ratio']:.3f}")
        else:
            output.append(f"📈 夏普比率:      尚未計算")

        if summary['last_updated']:
            output.append(f"\n⏱️  最後更新:      {summary['last_updated']}")

        output.append("")
        output.append("💡 提示: 綜合評分考慮了回測準確度、實時表現、勝率和風險調整後收益")

        return "\n".join(output)


def get_model_accuracy_display(symbol):
    """
    快速獲取模型準確度顯示（用於signal scripts）

    Parameters:
    -----------
    symbol : str
        股票代號

    Returns:
    --------
    str: 格式化的準確度信息
    """
    score, source = get_model_accuracy_score(symbol)
    if score is None:
        return "⚪ 模型品質(歷史驗證集命中率): 尚無數據"

    if score >= 70:
        indicator = "🟢"
    elif score >= 50:
        indicator = "🟡"
    else:
        indicator = "🔴"

    text = f"{indicator} 模型品質(歷史驗證集命中率): {score:.1f}%"

    if source and "proxy:" in source:
        proxy_symbol = source.split("proxy:", 1)[1]
        text += f" (proxy:{proxy_symbol})"
    return text


def get_model_accuracy_score(symbol):
    """
    取得模型準確度分數 (0-100) 與來源。
    回傳: (score_or_none, source_text_or_none)
    """
    tracker = ModelAccuracyTracker(symbol)
    summary = tracker.get_accuracy_summary()

    if summary['overall_score'] is not None:
        return float(summary['overall_score']), "overall"
    if summary.get('backtest_accuracy') is not None:
        return float(summary['backtest_accuracy']), "backtest"

    # Some legacy/acquired tickers do not have standalone accuracy history.
    proxy_symbol_map = {
        # Keep empty by default. Proxy fallback can hide missing real data.
        # Add mappings only when explicitly needed.
    }
    proxy_symbol = proxy_symbol_map.get(symbol.upper())
    if proxy_symbol:
        proxy_tracker = ModelAccuracyTracker(proxy_symbol)
        proxy_summary = proxy_tracker.get_accuracy_summary()
        if proxy_summary['overall_score'] is not None:
            return float(proxy_summary['overall_score']), f"overall-proxy:proxy:{proxy_symbol}"
        if proxy_summary.get('backtest_accuracy') is not None:
            return float(proxy_summary['backtest_accuracy']), f"backtest-proxy:proxy:{proxy_symbol}"

    # Fallback: read from feature_importance.json
    import glob
    clean_symbol = symbol.replace('.', '_')
    patterns = [
        str(Path(__file__).parent / f"{symbol}_feature_importance.json"),
        str(Path(__file__).parent / f"{clean_symbol}_feature_importance.json"),
    ]
    if proxy_symbol:
        proxy_clean = proxy_symbol.replace('.', '_')
        patterns.extend([
            str(Path(__file__).parent / f"{proxy_symbol}_feature_importance.json"),
            str(Path(__file__).parent / f"{proxy_clean}_feature_importance.json"),
        ])

    for pattern in patterns:
        for fi_file in glob.glob(pattern):
            try:
                with open(fi_file, 'r', encoding='utf-8') as f:
                    fi_data = json.load(f)
                acc = fi_data.get('model_accuracy')
                if acc is not None:
                    score = float(acc) * 100
                    source = "feature_importance"
                    if proxy_symbol and proxy_symbol in fi_file:
                        source = f"feature_importance-proxy:proxy:{proxy_symbol}"
                    return score, source
            except Exception:
                pass

    return None, None


def should_mute_ai_signal(symbol, threshold=52):
    """
    當模型準確度低於 threshold 時回傳 True，表示應靜音 AI 動作信號。
    """
    score, _ = get_model_accuracy_score(symbol)
    return score is not None and score < threshold


# 使用範例
if __name__ == "__main__":
    # 示例1: 更新訓練統計
    tracker = ModelAccuracyTracker("2330.TW")
    tracker.update_training_stats(
        training_acc=85.5,
        validation_acc=78.3,
        backtest_acc=72.1,
        win_rate=65.4,
        sharpe_ratio=1.85
    )

    # 示例2: 顯示準確度
    print(tracker.format_accuracy_display())
    print("\n")

    # 示例3: 記錄預測
    tracker.record_prediction("BUY", actual_result=True)

    # 示例4: 快速顯示
    print(get_model_accuracy_display("2330.TW"))
