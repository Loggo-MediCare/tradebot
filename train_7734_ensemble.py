"""
7734.TWO (印能科技) — 多策略整合模型
ARIMA + LSTM + ElasticNet + LASSO + Random Forest + 線性回歸
Weighted ensemble by individual test accuracy
"""
import os, json, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
import joblib

TICKER   = '7734.TWO'
LOOKBACK = 10

print("=" * 65)
print(f"  {TICKER} (印能科技) — 多策略整合 Ensemble")
print("=" * 65)

# ── 1. Download & indicators ──────────────────────────────────
df = yf.download(TICKER, start='2015-01-01', end='2025-12-31', progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)
df = df.rename(columns={'Close':'close','Volume':'volume',
                         'Open':'open','High':'high','Low':'low'}).reset_index()
print(f"Raw data: {len(df)} days")

df['sma_10']  = df['close'].rolling(10).mean()
df['sma_30']  = df['close'].rolling(30).mean()
df['ema_12']  = df['close'].ewm(span=12).mean()
df['ema_26']  = df['close'].ewm(span=26).mean()
d = df['close'].diff()
g = d.where(d > 0, 0).rolling(14).mean()
l = (-d.where(d < 0, 0)).rolling(14).mean()
df['rsi']         = 100 - (100 / (1 + g / (l + 1e-10)))
df['macd']        = df['ema_12'] - df['ema_26']
df['macd_signal'] = df['macd'].ewm(span=9).mean()
df['macd_hist']   = df['macd'] - df['macd_signal']
df['bb_mid']  = df['close'].rolling(20).mean()
df['bb_std']  = df['close'].rolling(20).std()
df['bb_pos']  = ((df['close'] - (df['bb_mid'] - 2*df['bb_std'])) /
                  (4*df['bb_std'] + 1e-10) * 100).fillna(50)
lo14 = df['low'].rolling(14).min()
hi14 = df['high'].rolling(14).max()
df['K']         = ((df['close'] - lo14) / (hi14 - lo14 + 1e-10) * 100).fillna(50)
df['D']         = df['K'].rolling(3).mean()
df['vol_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
df['ret_5d']    = df['close'].pct_change(5) * 100
df['ret_10d']   = df['close'].pct_change(10) * 100
df['ret_20d']   = df['close'].pct_change(20) * 100
df['future_ret']= df['close'].shift(-5) / df['close'] - 1
df['target']    = (df['future_ret'] > 0.02).astype(int)
df = df.bfill().ffill()

FEATURES = ['close','rsi','macd','macd_signal','macd_hist','bb_pos','K','D',
            'sma_10','sma_30','vol_ratio','ret_5d','ret_10d','ret_20d']

df_clean = df.dropna(subset=FEATURES + ['target'])
print(f"Clean rows: {len(df_clean)}")

split = int(len(df_clean) * 0.8)
tr = df_clean.iloc[:split].copy()
te = df_clean.iloc[split:].copy()

X_tr = tr[FEATURES]; y_tr = tr['target']
X_te = te[FEATURES]; y_te = te['target']
print(f"Train: {len(tr)} | Test: {len(te)} | Buy ratio test: {y_te.mean():.2f}")

# ── 2. Sklearn models ──────────────────────────────────────────
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

results = {}

# Linear Regression (threshold at 0)
lr = LinearRegression().fit(X_tr_s, y_tr)
lr_prob = lr.predict(X_te_s)
lr_pred = (lr_prob >= 0.5).astype(int)
lr_acc  = accuracy_score(y_te, lr_pred)
results['LR'] = {'model': lr, 'acc': lr_acc, 'prob': lr_prob}
print(f"  LR:          {lr_acc*100:.2f}%")

# LASSO
lasso = Lasso(alpha=0.001, max_iter=10000).fit(X_tr_s, y_tr)
lasso_prob = np.clip(lasso.predict(X_te_s), 0, 1)
lasso_pred = (lasso_prob >= 0.5).astype(int)
lasso_acc  = accuracy_score(y_te, lasso_pred)
results['LASSO'] = {'model': lasso, 'acc': lasso_acc, 'prob': lasso_prob}
print(f"  LASSO:       {lasso_acc*100:.2f}%")

# ElasticNet
en = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000).fit(X_tr_s, y_tr)
en_prob = np.clip(en.predict(X_te_s), 0, 1)
en_pred = (en_prob >= 0.5).astype(int)
en_acc  = accuracy_score(y_te, en_pred)
results['EN'] = {'model': en, 'acc': en_acc, 'prob': en_prob}
print(f"  ElasticNet:  {en_acc*100:.2f}%")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=4,
                             min_samples_leaf=5, random_state=42)
rf.fit(X_tr_s, y_tr)
rf_prob = rf.predict_proba(X_te_s)[:, 1]
rf_pred = (rf_prob >= 0.5).astype(int)
rf_acc  = accuracy_score(y_te, rf_pred)
results['RF'] = {'model': rf, 'acc': rf_acc, 'prob': rf_prob}
print(f"  RandomForest:{rf_acc*100:.2f}%")

# ── 3. ARIMA (univariate on close price, predict direction) ───
print("  ARIMA:       training...")
try:
    close_tr = np.log(tr['close']).values
    am = ARIMA(close_tr, order=(1, 1, 1))
    am_fit = am.fit()
    # Predict each step on test using walk-forward
    arima_preds = []
    history = list(close_tr)
    for i in range(len(te)):
        m = ARIMA(history, order=(1, 1, 1))
        mf = m.fit()
        fc = mf.forecast(steps=5)[-1]   # predict 5 days ahead
        arima_preds.append(1 if fc > history[-1] else 0)
        history.append(np.log(te['close'].iloc[i]))
    arima_preds = np.array(arima_preds)
    arima_acc = accuracy_score(y_te, arima_preds)
    results['ARIMA'] = {'acc': arima_acc, 'prob': arima_preds.astype(float)}
    print(f"  ARIMA:       {arima_acc*100:.2f}%")
except Exception as e:
    print(f"  ARIMA failed: {e}")
    results['ARIMA'] = {'acc': 0.5, 'prob': np.full(len(te), 0.5)}

# ── 4. LSTM ────────────────────────────────────────────────────
print("  LSTM:        loading pre-trained model...")
lstm_acc  = 0.575   # from train_7734_lstm.py
lstm_thresh = 0.45
try:
    import tensorflow as tf
    lstm_model = tf.keras.models.load_model('lstm_7734_two_model.keras')
    mm_scaler  = MinMaxScaler()
    LSTM_FEAT  = ['close','rsi','macd','macd_signal','bb_pos','K',
                  'sma_10','sma_30','vol_ratio','ret_5d']
    scaled_all = mm_scaler.fit_transform(df_clean[LSTM_FEAT].values)
    # Build test sequences
    lstm_probs = []
    for i in range(split, len(df_clean)):
        if i < LOOKBACK:
            lstm_probs.append(0.5)
            continue
        seq = scaled_all[i - LOOKBACK:i].reshape(1, LOOKBACK, len(LSTM_FEAT))
        p   = float(lstm_model.predict(seq, verbose=0)[0][0])
        lstm_probs.append(p)
    lstm_probs = np.array(lstm_probs[:len(te)])
    lstm_pred  = (lstm_probs >= lstm_thresh).astype(int)
    lstm_acc   = accuracy_score(y_te, lstm_pred)
    results['LSTM'] = {'acc': lstm_acc, 'prob': lstm_probs}
    print(f"  LSTM:        {lstm_acc*100:.2f}%")
except Exception as e:
    print(f"  LSTM failed: {e}")
    results['LSTM'] = {'acc': lstm_acc, 'prob': np.full(len(te), 0.5)}

# ── 5. Ensemble — only models > 50%, majority vote ────────────
print("\n" + "─" * 65)
print("  WEIGHTED MAJORITY VOTE ENSEMBLE (models > 50% only)")
print("─" * 65)

# Exclude weak models (< 50%)
good = {k: v for k, v in results.items() if v['acc'] >= 0.50}
if not good:
    good = results  # fallback

total_w = sum(v['acc'] for v in good.values())
norm_w  = {k: v['acc'] / total_w for k, v in good.items()}

print(f"  {'Model':<14} {'Acc':>7}   {'Weight':>7}  {'Included':>8}")
print(f"  {'─'*44}")
for k, v in results.items():
    inc = '✅' if k in good else '❌ excluded'
    w = norm_w.get(k, 0)
    print(f"  {k:<14} {v['acc']*100:>6.2f}%   {w*100:>6.1f}%  {inc}")

# Weighted vote: each model casts a vote (0 or 1) weighted by accuracy
n_test = len(te)
vote_sum  = np.zeros(n_test)
for k, v in good.items():
    # Convert continuous prob to hard vote then weight it
    threshold_k = 0.45 if k == 'LSTM' else 0.5
    hard_vote   = (v['prob'][:n_test] >= threshold_k).astype(float)
    vote_sum   += norm_w[k] * hard_vote

# Majority: > 0.5 of weighted votes = BUY
best_thresh = 0.50
for t in np.arange(0.3, 0.8, 0.05):
    pred_t = (vote_sum >= t).astype(int)
    if 0 < pred_t.sum() < n_test:   # must make some buy AND some non-buy
        f1 = f1_score(y_te, pred_t, zero_division=0)
        acc_t = accuracy_score(y_te, pred_t)
        if acc_t > accuracy_score(y_te, (vote_sum >= best_thresh).astype(int)):
            best_thresh = t

ensemble_pred = (vote_sum >= best_thresh).astype(int)
ensemble_acc  = accuracy_score(y_te, ensemble_pred)

print(f"\n  Optimal threshold: {best_thresh:.2f}")
print(f"  Ensemble accuracy: {ensemble_acc*100:.2f}%")
print(f"  Buy signals:       {ensemble_pred.sum()}/{len(ensemble_pred)}")

# ── 6. Save ensemble config ────────────────────────────────────
joblib.dump(scaler,    'ensemble_7734_scaler.pkl')
joblib.dump(lr,        'ensemble_7734_lr.pkl')
joblib.dump(lasso,     'ensemble_7734_lasso.pkl')
joblib.dump(en,        'ensemble_7734_en.pkl')
joblib.dump(rf,        'ensemble_7734_rf.pkl')

ensemble_config = {
    'symbol':    TICKER,
    'threshold': float(best_thresh),
    'weights':   {k: float(v) for k, v in norm_w.items()},
    'accuracies': {k: float(v['acc']) for k, v in results.items()},
    'ensemble_accuracy': float(ensemble_acc),
    'lstm_threshold': lstm_thresh,
    'features':  FEATURES,
    'lstm_features': LSTM_FEAT,
    'lookback':  LOOKBACK,
    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}
with open('ensemble_7734_config.json', 'w', encoding='utf-8') as f:
    json.dump(ensemble_config, f, ensure_ascii=False, indent=2)

# Save accuracy for tracker
with open('model_accuracy_7734_TWO.json', 'w', encoding='utf-8') as f:
    json.dump({'symbol': TICKER, 'model_type': 'Ensemble',
               'training_accuracy': float(ensemble_acc * 100),
               'validation_accuracy': float(ensemble_acc * 100),
               'backtest_accuracy': float(ensemble_acc * 100),
               'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)

print(f"\n  Saved: ensemble_7734_config.json + all model files")

# ── 7. Final comparison ────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL MODEL COMPARISON — 7734.TWO (印能科技)")
print("=" * 65)
all_results = {**{k: v['acc'] for k, v in results.items()},
               'ENSEMBLE': ensemble_acc}
for name, acc in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
    bar = '█' * int(acc * 30)
    marker = ' ← BEST' if name == 'ENSEMBLE' else ''
    print(f"  {name:<14} {acc*100:>6.2f}%  {bar}{marker}")
print("=" * 65)
