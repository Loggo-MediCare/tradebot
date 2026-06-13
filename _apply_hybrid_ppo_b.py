"""
Apply Hybrid PPO to Template-B signal scripts
(the simpler template that uses MODEL_FILE + pred/buy_prob instead of action_value).
"""
import os, re, ast

BASE = os.path.dirname(os.path.abspath(__file__))

STOCKS_B = [
    'aaoi', 'amat', 'bax', 'cien', 'coin',
    'gpn',  'grmn', 'moh', 'mrna', 'omc', 'tpl',
]

PPO_BLOCK = '''
    # ── PPO second opinion ───────────────────────────────────────────────────
    ppo_model = None
    ppo_action = 0.0
    ppo_signal_label = 'N/A'
    try:
        from stable_baselines3 import PPO as _PPO
        ppo_model = _PPO.load(PPO_FILE)
        p_price = current_price
        obs = __import__('numpy').array([
            0, 100000, p_price,
            float(latest.get('sma_10', p_price)),
            float(latest.get('sma_30', p_price)),
            float(latest.get('sma_50', p_price)),
            float(latest.get('rsi',   50)),
            float(latest.get('macd',   0)),
            float(latest.get('macd_signal', 0)),
            float(latest.get('bb_upper', p_price)),
            float(latest.get('bb_lower', p_price)),
            float(latest.get('volume',  0)),
            0, 1.0, 1.0,
        ], dtype='float32')
        ppo_act, _ = ppo_model.predict(obs, deterministic=True)
        ppo_action = float(ppo_act[0])
        ppo_signal_label = 'BUY' if ppo_action > 0.3 else ('SELL' if ppo_action < -0.3 else 'HOLD')
        print(f"\\n🤖 PPO Action: {ppo_action:+.4f}  → {ppo_signal_label}")
    except Exception as _ppo_err:
        print(f"\\n⚠️  PPO model not available: {_ppo_err}")
    # ── End PPO ──────────────────────────────────────────────────────────────
'''


def convert_b(suffix):
    fname = f'get_trading_signal_{suffix}.py'
    fpath = os.path.join(BASE, fname)
    if not os.path.exists(fpath):
        print(f'  SKIP (no file): {fname}')
        return False

    raw = open(fpath, 'rb').read()
    if raw.startswith(b'\xef\xbb\xbf'):
        raw = raw[3:]
    content = raw.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')

    if 'ppo_model' in content or 'PPO_FILE' in content:
        print(f'  SKIP (already hybrid): {fname}')
        return False

    original = content

    # 1. Update docstring
    content = content.replace(
        'XGBoost 交易信號生成器',
        'Hybrid PPO + XGBoost 交易信號生成器',
        1
    )
    # fallback for Chinese variant
    content = content.replace(
        'XGBoost 模型生成今日交易策略',
        'Hybrid PPO + XGBoost 模型生成今日交易策略',
        1
    )

    # 2. Add PPO_FILE constant after MODEL_FILE line
    content = re.sub(
        r"(MODEL_FILE\s*=\s*'xgb_\w+_model\.pkl')",
        r"\1\nPPO_FILE   = 'ppo_" + suffix + r"_improved'",
        content,
        count=1
    )

    # 3. After the XGB predict block, insert PPO block.
    #    Find the last predict line + buy_prob assignment and insert after it.
    #    Pattern: "buy_prob = proba[1] * 100" OR "buy_prob = prediction_proba[1] * 100"
    insert_marker = re.search(
        r'(buy_prob\s*=\s*(?:proba|prediction_proba)\[1\]\s*\*\s*100[^\n]*\n)',
        content
    )
    if insert_marker:
        end = insert_marker.end()
        content = content[:end] + PPO_BLOCK + content[end:]
    else:
        print(f'  WARN: buy_prob line not found in {fname}')

    if content == original:
        print(f'  WARN: no changes made to {fname}')
        return False

    try:
        ast.parse(content)
    except SyntaxError as e:
        print(f'  ERR syntax: {fname}: {e}')
        return False

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'  OK: {fname}')
    return True


ok = warn = 0
for suffix in STOCKS_B:
    if convert_b(suffix):
        ok += 1
    else:
        warn += 1

print(f'\nDone: {ok} converted, {warn} skipped/warned')
