"""
Bulk-apply Hybrid PPO + XGBoost to all listed signal scripts.
Skips files already converted (looks for 'from stable_baselines3 import PPO').
"""
import os, re, ast

BASE = os.path.dirname(os.path.abspath(__file__))

STOCKS = [
    # (suffix,  display_ticker)
    ('aaoi',  'AAOI'),
    ('amat',  'AMAT'),
    ('amd',   'AMD'),
    ('amzn',  'AMZN'),
    ('arm',   'ARM'),
    ('axon',  'AXON'),
    ('bax',   'BAX'),
    ('cien',  'CIEN'),
    ('coin',  'COIN'),
    ('gild',  'GILD'),
    ('googl', 'GOOGL'),
    ('gpn',   'GPN'),
    ('grmn',  'GRMN'),
    ('hsai',  'HSAI'),
    ('invz',  'INVZ'),
    ('ionq',  'IONQ'),
    ('jazz',  'JAZZ'),
    ('meta',  'META'),
    ('moh',   'MOH'),
    ('mpwr',  'MPWR'),
    ('mrna',  'MRNA'),
    ('msft',  'MSFT'),
    ('omc',   'OMC'),
    ('rdw',   'RDW'),
    ('smci',  'SMCI'),
    ('tpl',   'TPL'),
    ('txn',   'TXN'),
]

PPO_IMPORT = 'from stable_baselines3 import PPO\n'

# ── Replacement strings ───────────────────────────────────────────────────────

OLD_MODEL_LOAD = '''\
    # 1. 加载 XGBoost 模型
    model_filename = "xgb_{suffix}_model.pkl"
    print(f"\\n📦 加载 XGBoost 模型: {model_filename}")

    try:
        model = joblib.load(model_filename)
        print("✅ XGBoost 模型加载成功!")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None'''

NEW_MODEL_LOAD = '''\
    # 1. 加载模型 (XGBoost + PPO Hybrid)
    print(f"\\n📦 加载 Hybrid 模型: XGBoost + PPO")
    try:
        model = joblib.load("xgb_{suffix}_model.pkl")
        print("✅ XGBoost 模型加载成功!")
    except Exception as e:
        print(f"❌ XGBoost 模型加载失败: {{e}}")
        return None

    ppo_model = None
    try:
        ppo_model = PPO.load("ppo_{suffix}_improved")
        print("✅ PPO 模型加载成功!")
    except Exception as e:
        print(f"⚠️  PPO 模型加载失败 (将退化为纯XGB): {{e}}")'''

OLD_INFERENCE = '''\
    # 5. 使用 XGBoost 模型预测
    print("\\n🧠 XGBoost 模型分析中...")
    latest_features = df[feature_columns].iloc[[-1]]  # 获取最新数据的特征

    # XGBoost 预测 (0=不买, 1=买入)
    prediction = model.predict(latest_features)[0]
    prediction_proba = model.predict_proba(latest_features)[0]

    # 将 XGBoost 预测转换为 action_value (-1.0 到 1.0)
    # prediction=1 (买入) -> action_value > 0
    # prediction=0 (不买) -> action_value < 0
    if prediction == 1:
        # 买入信号，根据概率决定强度
        action_value = float(prediction_proba[1] * 2 - 1)  # 转换为 0 到 1 之间
        action_value = max(0.1, action_value)  # 确保至少 0.1
    else:
        # 不买/卖出信号
        action_value = -float(prediction_proba[0] * 2 - 1)  # 转换为 -1 到 0 之间
        action_value = min(-0.1, action_value)  # 确保至少 -0.1

    print(f"   XGBoost 预测: {\'买入\' if prediction == 1 else \'不买/观望\'}")
    print(f"   买入概率: {prediction_proba[1]*100:.2f}%")
    print(f"   Action Value: {action_value:+.4f}")'''

NEW_INFERENCE = '''\
    # 5a. XGBoost 推理
    print("\\n🧠 XGBoost 模型分析中...")
    latest_features = df[feature_columns].iloc[[-1]]
    prediction = model.predict(latest_features)[0]
    prediction_proba = model.predict_proba(latest_features)[0]

    if prediction == 1:
        xgb_action = float(prediction_proba[1] * 2 - 1)
        xgb_action = max(0.1, xgb_action)
    else:
        xgb_action = -float(prediction_proba[0] * 2 - 1)
        xgb_action = min(-0.1, xgb_action)

    print(f"   XGBoost 预测: {\'买入\' if prediction == 1 else \'不买/观望\'}")
    print(f"   买入概率: {prediction_proba[1]*100:.2f}%")
    print(f"   XGB Action: {{xgb_action:+.4f}}")

    # 5b. PPO 推理
    ppo_action = 0.0
    if ppo_model is not None:
        print("\\n🤖 PPO 模型分析中...")
        try:
            p = float(latest_data[\'close\'])
            obs = np.array([
                0, 100000, p,
                float(latest_data.get(\'sma_10\', p)),
                float(latest_data.get(\'sma_30\', p)),
                float(latest_data.get(\'sma_50\', p)),
                float(latest_data.get(\'rsi\', 50)),
                float(latest_data.get(\'macd\', 0)),
                float(latest_data.get(\'macd_signal\', 0)),
                float(latest_data.get(\'bb_upper\', p)),
                float(latest_data.get(\'bb_lower\', p)),
                float(latest_data.get(\'volume\', 0)),
                0, 1.0, 1.0,
            ], dtype=np.float32)
            ppo_act, _ = ppo_model.predict(obs, deterministic=True)
            ppo_action = float(ppo_act[0])
            ppo_signal = \'BUY\' if ppo_action > 0.3 else (\'SELL\' if ppo_action < -0.3 else \'HOLD\')
            print(f"   PPO Action: {{ppo_action:+.4f}}  → {{ppo_signal}}")
        except Exception as e:
            print(f"   ⚠️  PPO 推理失败: {{e}}")

    # 5c. Hybrid 融合 (60% XGB + 40% PPO)
    if ppo_model is not None:
        action_value = 0.6 * xgb_action + 0.4 * ppo_action
        print(f"\\n🔀 Hybrid Action: {{action_value:+.4f}}  (XGB×0.6 + PPO×0.4)")
    else:
        action_value = xgb_action
        print(f"\\n🔀 Action Value (XGB only): {{action_value:+.4f}}")'''


def convert_file(suffix, ticker):
    fname = f'get_trading_signal_{suffix}.py'
    fpath = os.path.join(BASE, fname)

    if not os.path.exists(fpath):
        print(f'  SKIP (no file): {fname}')
        return False

    raw = open(fpath, 'rb').read()
    if raw.startswith(b'\xef\xbb\xbf'):
        raw = raw[3:]
    content = raw.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')

    # Already converted?
    if 'from stable_baselines3 import PPO' in content:
        print(f'  SKIP (already hybrid): {fname}')
        return False

    original = content

    # 1. Add PPO import after 'warnings.filterwarnings'
    content = content.replace(
        "import warnings\nwarnings.filterwarnings('ignore')",
        "import warnings\nwarnings.filterwarnings('ignore')\n\nfrom stable_baselines3 import PPO",
        1
    )

    # 2. Update docstring
    content = content.replace(
        '使用训练好的 XGBoost 模型生成今日交易策略',
        '使用训练好的 Hybrid PPO + XGBoost 模型生成今日交易策略',
        1
    )

    # 3. Update print title (e.g. "🤖 美股 AMD AI 交易信号生成器")
    content = re.sub(
        r'(print\("🤖 美股 \w+ AI )交易信号生成器("\))',
        r'\1AI HYBRID PPO + XGBoost 交易信号生成器\2',
        content,
        count=1
    )

    # 4. Replace model loading section
    old_load = OLD_MODEL_LOAD.replace('{suffix}', suffix)
    new_load = NEW_MODEL_LOAD.replace('{suffix}', suffix)
    if old_load in content:
        content = content.replace(old_load, new_load, 1)
    else:
        print(f'  WARN: model-load pattern not found in {fname}')

    # 5. Replace inference section
    if OLD_INFERENCE in content:
        content = content.replace(OLD_INFERENCE, NEW_INFERENCE, 1)
    else:
        print(f'  WARN: inference pattern not found in {fname}')

    if content == original:
        print(f'  WARN: no changes made to {fname}')
        return False

    # Syntax check before writing
    try:
        ast.parse(content)
    except SyntaxError as e:
        print(f'  ERR syntax after transform: {fname}: {e}')
        return False

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'  OK: {fname}')
    return True


ok = warn = 0
for suffix, ticker in STOCKS:
    result = convert_file(suffix, ticker)
    if result:
        ok += 1
    else:
        warn += 1

print(f'\nDone: {ok} converted, {warn} skipped/warned')
