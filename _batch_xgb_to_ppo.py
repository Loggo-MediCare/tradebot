"""
Batch converter: XGBoost → PPO for 62 Taiwan signal scripts
Handles Type A (warnings.filterwarnings anchor) and Type B (MODEL_FILE constant) files
"""
import os, re, sys
sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STOCK_PPO_MAP = {
    '1301': 'ppo_1301_tw_improved',
    '1326': 'ppo_1326_tw_improved',
    '1560': 'ppo_1560_tw_improved',
    '1569': 'ppo_1569_two_improved',
    '1595': 'ppo_1595_two_improved',
    '2357': 'ppo_2357_tw_improved',
    '2404': 'ppo_2404_tw_improved',
    '2426': 'ppo_2426_tw_improved',
    '2485': 'ppo_2485_tw_improved',
    '2486': 'ppo_2486_tw_improved',
    '3004': 'ppo_3004_tw_improved',
    '3135': 'ppo_3135_tw_improved',
    '3163': 'ppo_3163_two_improved',
    '3363': 'ppo_3363_two_improved',
    '3380': 'ppo_3380_tw_improved',
    '3455': 'ppo_3455_two_improved',
    '3535': 'ppo_3535_tw_improved',
    '3563': 'ppo_3563_tw_improved',
    '3576': 'ppo_3576_tw_improved',
    '3583': 'ppo_3583_tw_improved',
    '3615': 'ppo_3615_two_improved',
    '3665': 'ppo_3665_tw_improved',
    '3680': 'ppo_3680_two_improved',
    '4564': 'ppo_4564_tw_improved',
    '4577': 'ppo_4577_two_improved',
    '4720': 'ppo_4720_tw_improved',
    '4768': 'ppo_4768_two_improved',
    '4772': 'ppo_4772_two_improved',
    '4951': 'ppo_4951_two_improved',
    '4989': 'ppo_4989_tw_improved',
    '4991': 'ppo_4991_two_improved',
    '5351': 'ppo_5351_two_improved',
    '6108': 'ppo_6108_tw_improved',
    '6138': 'ppo_6138_two_improved',
    '6139': 'ppo_6139_tw_improved',
    '6176': 'ppo_6176_tw_improved',
    '6187': 'ppo_6187_two_improved',
    '6213': 'ppo_6213_tw_improved',
    '6220': 'ppo_6220_two_improved',
    '6230': 'ppo_6230_tw_improved',
    '6234': 'ppo_6234_two_improved',
    '6438': 'ppo_6438_tw_improved',
    '6526': 'ppo_6526_tw_improved',
    '6531': 'ppo_6531_tw_improved',
    '6605': 'ppo_6605_tw_improved',
    '6672': 'ppo_6672_tw_improved',
    '6788': 'ppo_6788_two_improved',
    '6789': 'ppo_6789_tw_improved',
    '6830': 'ppo_6830_tw_improved',
    '6877': 'ppo_6877_two_improved',
    '6937': 'ppo_6937_tw_improved',
    '7703': 'ppo_7703_two_improved',
    '7751': 'ppo_7751_two_improved',
    '8027': 'ppo_8027_two_improved',
    '8028': 'ppo_8028_tw_improved',
    '8064': 'ppo_8064_two_improved',
    '8069': 'ppo_8069_two_improved',
    '8071': 'ppo_8071_two_improved',
    '8103': 'ppo_8103_tw_improved',
    '8147': 'ppo_8147_two_improved',
    '8438': 'ppo_8438_tw_improved',
    '8927': 'ppo_8927_two_improved',
}

# ── ImprovedTradingEnv class body ─────────────────────────────────────────────
IMPROVED_ENV_CLASS = '''
class ImprovedTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(ImprovedTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0
        self.total_trades = 0
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        current_price = float(row['close'])
        total_value = self.balance + self.shares_held * current_price
        stock_ratio = (self.shares_held * current_price) / total_value if total_value > 0 else 0
        cash_ratio = self.balance / total_value if total_value > 0 else 1
        obs = np.array([
            float(self.shares_held), float(self.balance), float(row['close']),
            float(row.get('sma_10', 0)), float(row.get('sma_30', 0)), float(row.get('sma_50', 0)),
            float(row.get('rsi', 50)), float(row.get('macd', 0)), float(row.get('macd_signal', 0)),
            float(row.get('bb_upper', 0)), float(row.get('bb_lower', 0)), float(row.get('volume', 0)),
            float(self.total_profit), float(stock_ratio), float(cash_ratio)
        ], dtype=np.float32)
        return obs

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = float(action[0])
        else:
            action = float(action)
        action = np.clip(action, -1.0, 1.0)
        current_price = float(self.df.iloc[self.current_step]['close'])
        old_total_value = self.balance + self.shares_held * current_price
        if action < -0.1:
            shares_to_sell = int(self.shares_held * abs(action))
            if shares_to_sell > 0:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                self.total_trades += 1
        elif action > 0.1:
            max_can_buy = int(self.balance // current_price)
            shares_to_buy = int(max_can_buy * action)
            if shares_to_buy > 0:
                self.balance -= shares_to_buy * current_price
                self.shares_held += shares_to_buy
                self.total_trades += 1
        new_total_value = self.balance + self.shares_held * current_price
        self.total_profit = new_total_value - self.initial_balance
        reward = self.total_profit / self.initial_balance
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_observation(), float(reward), done, False, {}
'''

# ── Type A: block injected after warnings.filterwarnings anchor ───────────────
ENV_CLASS_BLOCK = '''
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

''' + IMPROVED_ENV_CLASS

# ── Type B: prediction replacement (replaces X=latest[...] ... return {...}) ──
TYPE_B_PPO_PRED = '''    # PPO 環境與預測
    env = ImprovedTradingEnv(df)
    env.current_step = len(df) - 1
    obs = env._get_observation()

    print("\\n🧠 PPO AI 模型分析中...")
    action, _ = model.predict(obs, deterministic=True)
    action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
    print(f"   PPO Action Value: {action_value:+.4f}")

    rsi = float(latest.get('rsi', 50)); macd = float(latest.get('macd', 0)); ms = float(latest.get('macd_signal', 0))
    s10 = float(latest.get('sma_10', 0)); s30 = float(latest.get('sma_30', 0))

    print("\\n" + "=" * 80 + "\\n📊 技術指標\\n" + "=" * 80)
    print(f"當前價格: NT${current_price:.2f}")
    print(f"RSI: {rsi:.2f}  " + ("[超買]" if rsi > 70 else "[超賣]" if rsi < 30 else "[中性]"))
    print(f"MACD: {macd:.4f}  " + ("[金叉]" if macd > ms else "[死叉]"))
    print(f"均線: SMA10={s10:.2f}, SMA30={s30:.2f}  " + ("[多頭]" if s10 > s30 else "[空頭]"))

    try:
        patterns = analyze_candlestick_patterns(df, days=5)
        print(format_pattern_output(patterns))
        print("\\n型態評分調整: " + f"{get_pattern_score_adjustment(patterns):+.1f}" + " 分")
    except Exception:
        pass

    print("\\n" + "=" * 80 + "\\n🎯 交易信號\\n" + "=" * 80)
    print(f"模型輸出動作值: {action_value:+.4f}")
    if action_value > 0.1:
        print("🟢 買入信號 (BUY)")
        print(f"   PPO 動作強度: {action_value:+.4f}")
        print(f"   建議買入價格: NT${current_price*0.995:.2f} - NT${current_price*1.005:.2f}")
    elif action_value < -0.1:
        print("🔴 不建議買入 (SELL/WAIT)")
        print(f"   PPO 動作強度: {action_value:+.4f}")
    else:
        print("🟡 持有 (HOLD)")
        print(f"   PPO 動作強度: {action_value:+.4f}")

    print("=" * 80)
    return {'ticker': TICKER, 'price': current_price, 'action_value': action_value, 'rsi': rsi, 'macd': macd}'''


def convert_type_a(content, code, ppo_model):
    """Type A: has warnings.filterwarnings + model_filename = 'xgb_...' inside function."""

    # 1. Remove xgboost + joblib imports
    content = re.sub(r'\n*import xgboost as xgb\n+', '\n', content)
    content = re.sub(r'\n*import joblib\n+', '\n', content)

    # 2. Inject ENV_CLASS_BLOCK after warnings.filterwarnings('ignore')
    anchor = "warnings.filterwarnings('ignore')"
    if anchor in content:
        pos = content.index(anchor) + len(anchor)
        content = content[:pos] + ENV_CLASS_BLOCK + content[pos:]
    else:
        print(f"    WARN: Type A anchor not found for {code}")

    # 3. Replace model loading
    model_load_pattern = re.compile(
        r'model_filename\s*=\s*["\'][^"\']+["\'].*?'
        r'model\s*=\s*joblib\.load\(model_filename\).*?'
        r'print\(["\']✅ XGBoost 模型加载成功!["\'].*?\)',
        re.DOTALL
    )
    new_load = f'''model_path = "{ppo_model}"
    print(f"\\n📦 加载 PPO 模型: {{model_path}}")
    try:
        model = PPO.load(model_path)
        print("✅ PPO 模型加载成功!")'''
    content = model_load_pattern.sub(new_load, content)

    # 4. Replace XGBoost prediction block
    xgb_pred_pattern = re.compile(
        r'#\s*4[^\n]*[Xx][Gg][Bb][Oo][Oo][Ss][Tt][^\n]*\n.*?'
        r'feature_columns\s*=\s*\[.*?\].*?'
        r'ai_muted\s*=\s*should_mute_ai_signal\(',
        re.DOTALL
    )
    ticker_sym = f"'{code}.TWO'" if '_two_' in ppo_model else f"'{code}.TW'"
    ppo_pred = f'''# 4. PPO 環境與預測
    env = ImprovedTradingEnv(df)
    env.current_step = len(df) - 1
    obs = env._get_observation()

    # 5. 使用 PPO 模型預測
    print("\\n🧠 PPO AI 模型分析中...")
    action, _ = model.predict(obs, deterministic=True)
    action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
    print(f"   PPO Action Value: {{action_value:+.4f}}")

    ai_muted = should_mute_ai_signal('''
    content = xgb_pred_pattern.sub(ppo_pred, content)

    # 5. Replace XGBoost text labels
    content = content.replace('XGBoost 模型分析中', 'PPO AI 模型分析中')
    content = content.replace('XGBoost 預測', 'PPO 預測')
    content = content.replace('加載 XGBoost 模型', '加載 PPO 模型')
    content = content.replace('XGBoost 模型加載', 'PPO 模型加載')
    content = content.replace('使用訓練好的 XGBoost 模型', '使用訓練好的 PPO 模型')
    content = content.replace('XGBoost 所需特征', 'PPO 所需特征')

    return content


def convert_type_b(content, code, ppo_model):
    """Type B: has MODEL_FILE constant + joblib.load(MODEL_FILE)."""

    # 1. Fix MODEL_FILE value
    content = re.sub(
        r"MODEL_FILE\s*=\s*['\"][^'\"]*xgb[^'\"]*['\"]",
        f"MODEL_FILE = '{ppo_model}'",
        content
    )

    # 2. Fix imports: handle combined or standalone joblib
    if 'import numpy as np, pandas as pd, yfinance as yf, joblib' in content:
        content = content.replace(
            'import numpy as np, pandas as pd, yfinance as yf, joblib',
            'import numpy as np, pandas as pd, yfinance as yf\nimport gymnasium as gym\nfrom gymnasium import spaces\nfrom stable_baselines3 import PPO'
        )
    else:
        content = re.sub(r'\n*import joblib\n+', '\n', content)
        # Add PPO imports after numpy import line
        content = re.sub(
            r'(import numpy as np[^\n]*\n)',
            r'\1import gymnasium as gym\nfrom gymnasium import spaces\nfrom stable_baselines3 import PPO\n',
            content,
            count=1
        )

    # Also remove xgboost import if present
    content = re.sub(r'\n*import xgboost as xgb\n+', '\n', content)

    # 3. Inject ImprovedTradingEnv class before def get_trading_signal():
    if 'class ImprovedTradingEnv' not in content:
        m = re.search(r'\ndef get_trading_signal\(', content)
        if m:
            content = content[:m.start()] + '\n' + IMPROVED_ENV_CLASS + '\n' + content[m.start():]
        else:
            print(f"    WARN: cannot find def get_trading_signal() in {code}")

    # 4. Replace model loading: joblib.load(MODEL_FILE) → PPO.load(MODEL_FILE)
    content = re.sub(
        r'model\s*=\s*joblib\.load\(MODEL_FILE\)',
        'model = PPO.load(MODEL_FILE)',
        content
    )

    # 5. Replace prediction block: from X = latest[...] through return {...}
    #    Handles both FEATURE_COLUMNS and feature_columns (local var)
    pred_pattern = re.compile(
        r'X\s*=\s*latest\[\w+\]\.values\.reshape\(1,\s*-1\).*?'
        r'return\s*\{[^}]+\}',
        re.DOTALL
    )
    content = pred_pattern.sub(TYPE_B_PPO_PRED, content)

    # 6. Update text labels
    content = content.replace('XGBoost 模型分析中', 'PPO AI 模型分析中')
    content = content.replace('XGBoost 預測', 'PPO 預測')
    content = content.replace('加載 XGBoost 模型', '加載 PPO 模型')
    content = content.replace('XGBoost 模型加載', 'PPO 模型加載')

    return content


def convert_file(code, ppo_model):
    fname = f"get_trading_signal_{code}.py"
    fpath = os.path.join(SCRIPT_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  SKIP (not found): {fname}")
        return False

    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip if already fully converted
    if 'from stable_baselines3 import PPO' in content or 'PPO.load' in content:
        print(f"  SKIP (already PPO): {fname}")
        return False

    # Skip if no XGBoost/joblib indicators
    if ('import xgboost as xgb' not in content
            and 'joblib.load' not in content
            and 'import joblib' not in content):
        print(f"  SKIP (no xgboost): {fname}")
        return False

    original = content

    # Detect file type and convert
    is_type_b = 'MODEL_FILE' in content and 'joblib.load(MODEL_FILE)' in content
    if is_type_b:
        content = convert_type_b(content, code, ppo_model)
        file_type = 'B'
    else:
        content = convert_type_a(content, code, ppo_model)
        file_type = 'A'

    if content == original:
        print(f"  WARN: no changes made to {fname} (type {file_type})")
        return False

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"  ✅ converted [{file_type}]: {fname} → {ppo_model}")
    return True


if __name__ == '__main__':
    print(f"Converting {len(STOCK_PPO_MAP)} signal files from XGBoost → PPO")
    print("=" * 60)
    converted = 0
    skipped = 0
    for code, ppo_model in STOCK_PPO_MAP.items():
        result = convert_file(code, ppo_model)
        if result:
            converted += 1
        else:
            skipped += 1
    print("=" * 60)
    print(f"Done: {converted} converted, {skipped} skipped")
