import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots

# ==============================================================================
# ⚠️ 1. 導入自定義物件 (Import Custom Objects)
# 務必替換為您實際的 RL 代理和環境類別的導入路徑
# 這些類別通常定義了模型的層次結構、狀態定義和交易邏輯
# ==============================================================================

# 假設您的自定義類別定義在 'rl_trading_strategy_1280_earn2.py' 中
# 注意: 直接導入會觸發訓練，所以暫時註解掉，使用下面的替代類別
# try:
#     from rl_trading_strategy_1280_earn2 import TradingAgent, TradingEnvironment
# except ImportError:
#     print("⚠️ 錯誤: 無法導入 TradingAgent 或 TradingEnvironment。請檢查導入路徑。")

# 定義回測所需的環境類別
class TradingAgent: pass

class TradingEnvironment:
    """
    真實的交易環境，使用實際價格數據進行買賣
    Actions: 0=Hold, 1=Buy, 2=Sell
    """
    def __init__(self, data, initial_balance):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0
        self.max_steps = len(data) - 1

        # 交易狀態
        self.cash = initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0

        # 獲取價格列（處理MultiIndex情況）
        if isinstance(self.data.columns, pd.MultiIndex):
            # 如果是MultiIndex，取第一個ticker的Close價格
            self.price_column = [col for col in self.data.columns if 'Close' in str(col)][0]
        else:
            self.price_column = 'Close'

    @property
    def current_profit(self):
        """計算當前總資產（現金 + 持股價值）"""
        current_price = self.data.iloc[self.current_step][self.price_column]
        return self.cash + (self.shares_held * current_price)

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        return self._get_state()

    def _get_state(self):
        """
        生成當前狀態向量（與訓練時相同的格式）
        使用sigmoid轉換的價格差異窗口
        """
        window_size = 10
        t = self.current_step
        n = window_size + 1

        if t >= len(self.data):
            return np.zeros(window_size)

        # 獲取價格窗口
        d = t - n + 1
        if d >= 0:
            block = self.data.iloc[d:t + 1][self.price_column].values
        else:
            # 如果在開始位置，用第一個價格填充
            first_price = self.data.iloc[0][self.price_column]
            block = np.concatenate([np.repeat(first_price, -d),
                                   self.data.iloc[0:t + 1][self.price_column].values])

        # 計算sigmoid(price_diff)
        res = []
        for i in range(n - 1):
            price_diff = block[i + 1] - block[i]
            sigmoid_val = 1 / (1 + np.exp(-price_diff))
            res.append(sigmoid_val)

        return np.array(res, dtype=np.float32)

    def step(self, action):
        """
        執行交易動作
        action: 0=Hold, 1=Buy, 2=Sell
        """
        prev_profit = self.current_profit
        current_price = self.data.iloc[self.current_step][self.price_column]
        reward = 0

        # Action 1: Buy (買入)
        if action == 1:
            # 用所有現金買入股票
            if self.cash > current_price:
                shares_to_buy = int(self.cash / current_price)
                cost = shares_to_buy * current_price
                self.shares_held += shares_to_buy
                self.cash -= cost
                self.total_shares_bought += shares_to_buy

        # Action 2: Sell (賣出)
        elif action == 2:
            # 賣出所有持股
            if self.shares_held > 0:
                revenue = self.shares_held * current_price
                self.cash += revenue
                self.total_shares_sold += self.shares_held
                self.shares_held = 0

        # Action 0: Hold (持有) - 不做任何事

        # 移動到下一步
        self.current_step += 1
        is_done = self.current_step >= self.max_steps

        # 如果結束，清算所有持股
        if is_done and self.shares_held > 0:
            final_price = self.data.iloc[-1][self.price_column]
            self.cash += self.shares_held * final_price
            self.shares_held = 0

        # 計算獎勵（資產變化）
        current_profit = self.current_profit
        reward = current_profit - prev_profit

        # 獲取新狀態
        new_state = self._get_state() if not is_done else np.zeros(10)

        info = {
            'cash': self.cash,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'total_profit': current_profit - self.initial_balance
        }

        return new_state, reward, is_done, info
        
custom_objects = {
    # 添加 MSE loss 函數以正確加載模型
    'mse': tf.keras.losses.MeanSquaredError(),
}

# ==============================================================================
# 2. 數據準備 (Data Preparation)
# ==============================================================================
TICKER = "^GSPC" # 假設是 S&P 500 的指數符號
START_DATE = "2025-01-01" # 使用2025年數據進行回測（未見數據）
END_DATE = "2025-12-12"

print(f"下載 {TICKER} 的測試數據 ({START_DATE} 到 {END_DATE})...")
try:
    # 下載數據
    test_data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    # ⚠️ 這裡需要添加您的特徵工程和數據預處理步驟，以產生 'state' 所需的格式
    print(f"數據加載完成: {len(test_data)} 點")
except Exception as e:
     print(f"下載數據失敗: {e}，將使用模擬數據")
     # 失敗時使用模擬數據，以確保程式碼運行
     test_data = pd.DataFrame(np.random.rand(100, 5), columns=['Open', 'High', 'Low', 'Close', 'Volume'])

# ==============================================================================
# 3. 加載模型 (Load Model)
# ==============================================================================
MODEL_FILE = 'my_trading_bot.h5'
INITIAL_BALANCE = 100000

print(f"正在加載模型: {MODEL_FILE}...")
try:
    model = load_model(
        MODEL_FILE, 
        custom_objects=custom_objects 
    )
    print("模型加載成功！")
except Exception as e:
    print(f"加載模型時發生錯誤。請確認 'custom_objects' 包含了所有必需的類別/函數。錯誤: {e}")
    exit() # 如果模型無法加載，則中止程式

# ==============================================================================
# 4. 運行回測模擬 (Run Backtesting Simulation)
# ==============================================================================

# 初始化測試環境
test_env = TradingEnvironment(test_data, initial_balance=INITIAL_BALANCE)

# 重置環境，獲取初始狀態
state = test_env.reset() 
done = False

print("\n開始運行交易模擬...")
step_count = 0
portfolio_history = []  # 記錄每一步的資金變化
action_history = []  # 記錄每一步的行動
q_values_history = []  # 記錄Q值

# Debug: 打印前幾步的狀態和Q值
debug_steps = 5

while not done:
    # 準備輸入: 將狀態從 (特徵數,) 轉換為 (1, 特徵數)
    state_input = np.reshape(state, (1, -1))

    # 模型預測: 獲取 Q 值
    # verbose=0 可以隱藏 predict 的進度輸出
    q_values = model.predict(state_input, verbose=0)

    # 選擇行動: 選擇 Q 值最高的行動 (貪婪策略)
    action = np.argmax(q_values[0])

    # Debug: 打印前幾步的詳細信息
    if step_count < debug_steps:
        print(f"\n步驟 {step_count}:")
        print(f"  狀態: {state}")
        print(f"  Q值: {q_values[0]}")
        print(f"  選擇動作: {action} ({'Hold' if action==0 else 'Buy' if action==1 else 'Sell'})")

    # 記錄當前資金和行動
    portfolio_history.append(test_env.current_profit)
    action_history.append(action)
    q_values_history.append(q_values[0])

    # 執行行動: 獲取下一狀態、報酬、是否結束、額外資訊
    next_state, reward, done, info = test_env.step(action)

    # 更新狀態
    state = next_state
    step_count += 1

    if step_count % 50 == 0 or done:
        # 簡易進度輸出
        print(f"步數: {step_count}/{len(test_data)}, 當前價值: ${test_env.current_profit:.2f}", end='\r')

# 記錄最終資金
portfolio_history.append(test_env.current_profit)


# ==============================================================================
# 5. 輸出最終結果 (Final Result)
# ==============================================================================
print("\n\n================================")
print("回測結果總結")
print("================================")
print(f"初始資金: ${INITIAL_BALANCE:,.2f}")
print(f"最終資金: ${test_env.current_profit:,.2f}")
print(f"現金餘額: ${test_env.cash:,.2f}")
print(f"持有股票: {test_env.shares_held} 股")
profit = test_env.current_profit - INITIAL_BALANCE
profit_pct = (profit / INITIAL_BALANCE) * 100
print(f"\n總利潤/虧損: ${profit:,.2f} ({profit_pct:.2f}%)")
print(f"總共買入: {test_env.total_shares_bought} 股")
print(f"總共賣出: {test_env.total_shares_sold} 股")
print("================================")

# ==============================================================================
# 6. 生成可視化圖表 (Generate Visualization)
# ==============================================================================
print("\n生成圖表...")

# 創建圖表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# 圖表1: 資金變化
ax1.plot(portfolio_history, linewidth=2, color='blue', label='Portfolio Value')
ax1.axhline(y=INITIAL_BALANCE, color='red', linestyle='--', linewidth=1, label='Initial Balance')
ax1.set_xlabel('Trading Steps', fontsize=12)
ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
ax1.set_title('Trading Bot Performance - Portfolio Value Over Time', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 添加最終資金註解
ax1.annotate(f'Final: ${test_env.current_profit:.2f}',
             xy=(len(portfolio_history)-1, test_env.current_profit),
             xytext=(10, 10), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# 圖表2: 行動分布
if len(action_history) > 0:
    action_counts = pd.Series(action_history).value_counts().sort_index()
    action_labels = {0: 'Hold', 1: 'Buy', 2: 'Sell'}  # 假設的行動標籤
    labels = [action_labels.get(i, f'Action {i}') for i in action_counts.index]

    ax2.bar(range(len(action_counts)), action_counts.values, color=['gray', 'green', 'red'][:len(action_counts)])
    ax2.set_xlabel('Action Type', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Trading Actions Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(action_counts)))
    ax2.set_xticklabels(labels)
    ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# 保存圖表
chart_filename = 'backtest_results.png'
plt.savefig(chart_filename, dpi=150, bbox_inches='tight')
print(f"圖表已保存到: {chart_filename}")

# 同時保存詳細數據到CSV
results_df = pd.DataFrame({
    'Step': range(len(portfolio_history)),
    'Portfolio_Value': portfolio_history
})
csv_filename = 'backtest_results.csv'
results_df.to_csv(csv_filename, index=False)
print(f"數據已保存到: {csv_filename}")

print("\n回測完成！")


# ==============================================================================
# 1. 導入自定義物件 (Import Custom Objects)
# ==============================================================================

# 假設您的 TradingAgent 和 TradingEnvironment 定義在 rl_trading_strategy_1280_earn2.py 中
# try:
#     # 這是實際的導入方式：
#     from rl_trading_strategy_1280_earn2 import TradingAgent, TradingEnvironment 
#     print("✅ 已成功導入 TradingAgent 和 TradingEnvironment。")
# except ImportError as e:
#     # 如果執行時遇到 ImportError，可能是因為您在運行程式碼時，
#     # 所在的目錄不是 C:\Users\Silvi\Projects\trading-bot。
#     # 如果出現錯誤，請確保您在正確的目錄下運行或調整 PYTHONPATH。
#     print(f"❌ 錯誤: 無法從 rl_trading_strategy_1280_earn2 導入所需類別。錯誤訊息: {e}")
#     # 以下為模擬類別的替代方案 (僅在導入失敗時使用)
#     class TradingAgent: pass
#     class TradingEnvironment:
#         # 必須定義 __init__, reset, step 方法供主程式碼調用
#         def __init__(self, data, initial_balance):
#             self.data = data
#             self.current_profit = initial_balance
#             self.current_step = 0
#             self.initial_balance = initial_balance
#             self.max_steps = len(data) - 1

#         def reset(self):
#             self.current_step = 0
#             return np.zeros(10)

#         def step(self, action):
#             self.current_step += 1
#             is_done = self.current_step >= self.max_steps
#             new_state = np.random.rand(10) if not is_done else np.zeros(10)
#             reward = 0.0
#             if is_done:
#                  self.current_profit += np.random.uniform(500, 10000)
#             return new_state, reward, is_done, {}