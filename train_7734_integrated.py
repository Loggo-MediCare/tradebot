"""
7734.TWO (印能科技) — Integrated Strategy: LASSO + ARIMAX + Technical Analysis
Adapted from IntegratedTradingStrategy for Taiwan stocks
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Technical indicator helpers
# =============================================================================
def calculate_MA(prices, period=10):
    prices = np.array(prices, dtype=float)
    ma = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        ma[i] = np.mean(prices[i - period + 1:i + 1])
    return ma

def calculate_EMA(prices, period=12):
    prices = np.array(prices, dtype=float)
    ema = np.full(len(prices), np.nan)
    if len(prices) < period:
        return ema
    multiplier = 2 / (period + 1)
    ema[period - 1] = np.mean(prices[:period])
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]
    return ema

def calculate_RSI(prices, period=14):
    prices = np.array(prices, dtype=float)
    rsi = np.full(len(prices), np.nan)
    if len(prices) < period + 1:
        return rsi
    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rsi[period] = 100 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))
    for i in range(period + 1, len(prices)):
        avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
        rsi[i] = 100 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))
    return rsi

def calculate_MACD(prices, fast=12, slow=26, signal=9):
    prices = np.array(prices, dtype=float)
    ema_f = calculate_EMA(prices, fast)
    ema_s = calculate_EMA(prices, slow)
    macd  = ema_f - ema_s
    sig   = np.full(len(prices), np.nan)
    start = slow - 1
    if len(macd) > start + signal:
        valid = macd[start:]
        sig_vals = calculate_EMA(valid, signal)
        sig[start:] = sig_vals
    return macd, sig, macd - sig


# =============================================================================
# IntegratedTradingStrategy — adapted for 7734.TWO
# =============================================================================
class IntegratedTradingStrategy:
    def __init__(self, target_stock='7734.TWO', data_period_years=2):
        self.target_stock    = target_stock
        # Use Taiwan tech peers as correlated assets
        self.correlated_stocks = [target_stock, '2330.TW', '3481.TW']
        self.return_period   = 5    # predict next 5-day return
        self.data_period_years = data_period_years
        self.ma_period       = 10   # shorter window for limited data
        self.arimax_model    = None
        self.linear_model    = None
        self.best_model_name = None
        self.scaler          = StandardScaler()

    def load_data(self, start_date, end_date):
        print(f"  Loading data for {self.target_stock}...")
        stk_data = yf.download(self.correlated_stocks,
                               start=start_date, end=end_date,
                               auto_adjust=True, progress=False)
        if isinstance(stk_data.columns, pd.MultiIndex):
            price_data = stk_data['Close'].copy()
        else:
            price_data = pd.DataFrame(stk_data['Close'])
            price_data.columns = [self.target_stock]
        price_data = price_data.ffill().bfill()
        return price_data

    def prepare_features(self, price_data):
        print("  Engineering features...")
        target = price_data[self.target_stock]

        # Target: log return over next 5 days
        Y = np.log(target).diff(self.return_period).shift(-self.return_period)
        Y.name = 'Target_Return'

        # Features: correlated stock returns
        X1 = np.log(price_data).diff(self.return_period)

        # Momentum features
        for p in [5, 10, 20]:
            col = np.log(target).diff(p)
            col.name = f'Mom_{p}d'
            X1 = pd.concat([X1, col], axis=1)

        # Volume ratio (from target ticker)
        raw = yf.download(self.target_stock,
                          start=price_data.index[0], end=price_data.index[-1],
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        vol_ratio = raw['Volume'] / (raw['Volume'].rolling(10).mean() + 1e-10)
        vol_ratio.name = 'Vol_Ratio'
        X1 = pd.concat([X1, vol_ratio], axis=1)

        dataset = pd.concat([Y, X1], axis=1).dropna()
        # Subsample to weekly to reduce serial correlation
        dataset = dataset.iloc[::self.return_period, :]

        X = dataset.drop(columns=['Target_Return'])
        Y = dataset['Target_Return']
        return X, Y, price_data

    def train_models(self, X_train, Y_train, X_test, Y_test):
        print("  Training LR / LASSO / ElasticNet / ARIMAX...")

        X_tr_s = self.scaler.fit_transform(X_train)
        X_te_s = self.scaler.transform(X_test)

        models = {
            'LR':    LinearRegression(),
            'LASSO': Lasso(alpha=0.001, max_iter=10000),
            'EN':    ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000),
        }

        results = {}
        best_mse = float('inf')
        for name, m in models.items():
            m.fit(X_tr_s, Y_train)
            pred = m.predict(X_te_s)
            mse  = mean_squared_error(Y_test, pred)
            # Direction accuracy (up/down correct)
            dir_acc = np.mean(np.sign(pred) == np.sign(Y_test))
            results[name] = {'mse': mse, 'dir_acc': dir_acc, 'model': m, 'pred': pred}
            print(f"    {name:<8} MSE={mse:.6f}  Dir-Acc={dir_acc*100:.1f}%")
            if mse < best_mse:
                best_mse = mse
                self.best_model_name = name
                self.linear_model    = m

        print(f"  Best linear model: {self.best_model_name}")

        # ARIMAX
        corrs = X_train.corrwith(Y_train).abs().sort_values(ascending=False)
        top3  = corrs.head(3).index.tolist()
        self.arimax_exog = top3
        arimax_dir_acc   = None
        try:
            am = ARIMA(endog=Y_train.values,
                       exog=X_train[top3].values,
                       order=(1, 0, 1))
            self.arimax_model = am.fit()
            ap = self.arimax_model.forecast(steps=len(X_test),
                                             exog=X_test[top3].values)
            am_mse = mean_squared_error(Y_test, ap)
            arimax_dir_acc = float(np.mean(np.sign(ap) == np.sign(Y_test)))
            print(f"    ARIMAX   MSE={am_mse:.6f}  Dir-Acc={arimax_dir_acc*100:.1f}%")
        except Exception as e:
            print(f"    ARIMAX failed: {e}")

        return results, arimax_dir_acc

    def analyze_technical(self, price_data):
        prices = price_data[self.target_stock].values
        ma    = calculate_MA(prices, self.ma_period)
        rsi   = calculate_RSI(prices, 14)
        macd, sig, hist = calculate_MACD(prices)

        score = 0
        curr_price = prices[-1]
        curr_ma    = ma[-1]
        curr_rsi   = rsi[-1]
        curr_hist  = hist[-1] if not np.isnan(hist[-1]) else 0

        if curr_price > curr_ma: score += 1
        else: score -= 1
        if curr_hist > 0: score += 1
        else: score -= 1
        if curr_rsi >= 60: score += 1
        elif curr_rsi <= 40: score -= 1

        return {
            'price': curr_price, 'ma': curr_ma,
            'rsi': curr_rsi, 'macd_hist': curr_hist,
            'tech_score': score,
            'ma_trend': 'BULL' if curr_price > curr_ma else 'BEAR'
        }

    def run_strategy(self):
        print("\n" + "=" * 60)
        print(f"  {self.target_stock} (印能科技) — Integrated Strategy")
        print("=" * 60)

        end_date   = datetime.now()
        start_date = end_date - timedelta(days=365 * self.data_period_years)

        price_data   = self.load_data(start_date, end_date)
        X, Y, raw    = self.prepare_features(price_data)

        print(f"  Samples after subsampling: {len(X)}")
        if len(X) < 20:
            print("  ⚠️  Too few samples for reliable training.")

        split = int(len(X) * 0.8)
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        Y_tr, Y_te = Y.iloc[:split], Y.iloc[split:]

        lin_results, arimax_dir = self.train_models(X_tr, Y_tr, X_te, Y_te)

        # Predict next period
        X_sc   = self.scaler.transform(X.iloc[-1:])
        pred_lr = float(self.linear_model.predict(X_sc)[0])

        pred_arimax = 0.0
        if self.arimax_model:
            try:
                pred_arimax = float(
                    self.arimax_model.forecast(
                        steps=1, exog=X.iloc[-1:][self.arimax_exog].values
                    ).iloc[0]
                )
            except Exception:
                pass

        avg_pred = (pred_lr + pred_arimax) / 2
        tech     = self.analyze_technical(raw)

        print(f"\n{'─'*60}")
        print(f"  AI FORECAST (next {self.return_period} days)")
        print(f"{'─'*60}")
        print(f"  {self.best_model_name:<8} prediction: {pred_lr*100:+.2f}%")
        print(f"  ARIMAX     prediction: {pred_arimax*100:+.2f}%")
        print(f"  Combined   forecast:   {avg_pred*100:+.2f}%")

        print(f"\n  TECHNICAL STATUS")
        print(f"  Price: NT${tech['price']:.2f}  MA{self.ma_period}: NT${tech['ma']:.2f}  ({tech['ma_trend']})")
        print(f"  RSI: {tech['rsi']:.1f}  MACD hist: {tech['macd_hist']:.4f}")
        print(f"  Tech score: {tech['tech_score']} / 3")

        print(f"\n{'─'*60}")
        print(f"  FINAL SIGNAL")
        print(f"{'─'*60}")

        if tech['tech_score'] >= 1 and avg_pred > 0.005:
            signal = "🟢 買入 (BUY) — AI + Technical aligned"
        elif avg_pred > 0.015 and tech['rsi'] > 30:
            signal = "🟡 潛伏買入 (ACCUMULATE) — AI bullish, technicals lagging"
        elif tech['tech_score'] <= -1 and avg_pred < -0.005:
            signal = "🔴 賣出 (SELL) — AI + Technical both bearish"
        else:
            signal = "⚪ 觀望 (WAIT) — No clear confluence"

        print(f"  {signal}")

        print(f"\n  ⚠️  Note: 印能科技 has only ~{len(price_data)} trading days of data.")
        print(f"  Models trained on ~{len(X_tr)} weekly samples. Treat with low confidence.")
        print("=" * 60)

        return {'pred_lr': pred_lr, 'pred_arimax': pred_arimax,
                'avg_pred': avg_pred, 'tech': tech, 'signal': signal}


if __name__ == "__main__":
    bot = IntegratedTradingStrategy(target_stock='7734.TWO', data_period_years=2)
    bot.run_strategy()
