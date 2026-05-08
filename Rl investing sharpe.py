"""
強化學習投資環境 - 使用滾動夏普比率 + 配置調整懲罰項
=====================================================

本程式實現了一個完整的 DQL（Deep Q-Learning）投資代理，
使用滾動夏普比率減去配置調整懲罰項作為獎勵函數。

核心概念：
1. 滾動夏普比率：考慮風險調整後的報酬
2. 配置調整懲罰項：避免過度交易，鼓勵穩定配置
3. 獎勵 = 夏普比率 - 懲罰項

作者：Claude AI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from collections import deque
import random
import math
import warnings
warnings.filterwarnings('ignore')

# 嘗試導入 TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("警告：TensorFlow 未安裝，將使用簡化版本")


# ============================================================================
# 第一部分：輔助類別定義
# ============================================================================

class ObservationSpace:
    """觀察空間類別"""
    def __init__(self, n):
        self.shape = (n,)


class ActionSpace:
    """
    動作空間類別
    
    動作為連續值，範圍 [0, 1]
    代表投資於風險資產的比例
    """
    def __init__(self, n):
        self.n = n
    
    def seed(self, seed):
        random.seed(seed)
    
    def sample(self):
        """隨機抽取一個動作（0到1之間的浮點數）"""
        return random.random()


# ============================================================================
# 第二部分：投資環境類別
# ============================================================================

class InvestingEnvironment:
    """
    投資環境 - 使用滾動夏普比率作為獎勵
    
    狀態空間（5維）：
    - Xt: 風險資產（股票）當前價格（標準化）
    - Yt: 無風險資產（債券）當前價格（標準化）
    - Xt - Yt: 兩種資產的價格差異
    - xt: 當前風險資產配置比例
    - yt: 當前無風險資產配置比例
    
    動作空間：
    - 連續值 [0, 1]：風險資產的配置比例
    
    獎勵函數：
    - reward = 滾動夏普比率 - 配置調整懲罰項
    """
    
    def __init__(self, S0=1.0, T=1.0, r_=None, mu_=None, sigma_=None,
                 steps=252, amount=1.0, rolling_window=20,
                 penalty_factor=1.0):
        """
        初始化投資環境
        
        參數：
        -----
        S0 : float - 初始資產價格（標準化為1）
        T : float - 投資期間（年）
        r_ : list - 無風險利率候選值
        mu_ : list - 風險資產漂移率候選值
        sigma_ : list - 風險資產波動率候選值
        steps : int - 總步數（交易日數）
        amount : float - 初始投資金額
        rolling_window : int - 計算滾動波動率的窗口大小
        penalty_factor : float - 配置調整懲罰係數
        """
        # 預設參數
        self.r_ = r_ if r_ is not None else [0.0, 0.025, 0.05]
        self.mu_ = mu_ if mu_ is not None else [0.05, 0.1, 0.15]
        self.sigma_ = sigma_ if sigma_ is not None else [0.1, 0.2, 0.3]
        
        # 環境設定
        self.initial_value = S0
        self.maturity = T
        self.steps = steps
        self.initial_balance = amount
        self.rolling_window = rolling_window
        self.penalty_factor = penalty_factor
        
        # 空間定義
        self.observation_space = ObservationSpace(5)
        self.osn = self.observation_space.shape[0]
        self.action_space = ActionSpace(1)
        
        # 初始化
        self._generate_data()
        self.portfolios = pd.DataFrame()
        self.episode = 0
    
    def _generate_data(self):
        """
        生成模擬資產價格數據
        
        使用幾何布朗運動（GBM）模擬風險資產
        無風險資產使用確定性增長
        """
        # 隨機選擇參數
        self.short_rate = random.choice(self.r_)
        self.index_drift = random.choice(self.mu_)
        self.volatility = random.choice(self.sigma_)
        
        self.dt = self.maturity / self.steps
        
        # 模擬風險資產價格（GBM）
        s = [self.initial_value]
        for t in range(1, self.steps + 1):
            drift = (self.index_drift - self.volatility ** 2 / 2) * self.dt
            diffusion = self.volatility * math.sqrt(self.dt) * random.gauss(0, 1)
            st = s[t - 1] * math.exp(drift + diffusion)
            s.append(st)
        
        # 建立數據框
        self.data = pd.DataFrame(s, columns=['Xt'])
        
        # 計算無風險資產價值
        self.data['Yt'] = self.initial_value * np.exp(
            self.short_rate * np.arange(len(self.data)) * self.dt
        )
    
    def _get_state(self):
        """
        獲取當前狀態
        
        返回：
        -----
        tuple : (state_array, info_dict)
        """
        Xt = self.data['Xt'].iloc[self.bar]
        Yt = self.data['Yt'].iloc[self.bar]
        
        # 狀態向量：[Xt, Yt, Xt-Yt, xt, yt]
        state = np.array([Xt, Yt, Xt - Yt, self.xt, self.yt])
        
        return state, {}
    
    def reset(self):
        """
        重置環境到初始狀態
        
        返回：
        -----
        tuple : (initial_state, info)
        """
        self.bar = 0
        self.xt = 0  # 風險資產配置比例
        self.yt = 0  # 無風險資產配置比例
        self.treward = 0
        
        self.portfolio_value = self.initial_balance
        self.portfolio_value_new = self.initial_balance
        
        self.episode += 1
        self._generate_data()
        
        # 重置收益記錄
        self.returns_history = []
        
        self.state, _ = self._get_state()
        return self.state, _
    
    def _calculate_rolling_sharpe(self):
        """
        計算滾動夏普比率（正規化版本）
        
        返回：
        -----
        float : 滾動夏普比率（縮放到合理範圍）
        """
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array(self.returns_history)
        
        # 使用最近 rolling_window 期的數據
        if len(returns) > self.rolling_window:
            returns = returns[-self.rolling_window:]
        
        # 計算平均報酬率（不年化，保持每日尺度）
        mean_return = np.mean(returns)
        
        # 計算波動率
        std_return = np.std(returns, ddof=1) if len(returns) > 1 else 0.01
        
        # 避免除以零，設置最小波動率
        if std_return < 0.001:
            std_return = 0.001
        
        # 夏普比率（保持每日尺度，不年化）
        # 這樣夏普比率會在合理範圍內（大約 -3 到 +3）
        sharpe = mean_return / std_return
        
        # 裁剪到合理範圍，避免極端值
        sharpe = np.clip(sharpe, -5, 5)
        
        return sharpe
    
    def _calculate_penalty(self, old_allocation, new_allocation):
        """
        計算配置調整懲罰項
        
        懲罰項 = penalty_factor * (old_allocation - new_allocation)^2
        
        參數：
        -----
        old_allocation : float - 先前的風險資產配置
        new_allocation : float - 新的風險資產配置
        
        返回：
        -----
        float : 懲罰值
        """
        return self.penalty_factor * (old_allocation - new_allocation) ** 2
    
    def add_results(self, pl, sharpe, penalty, reward):
        """
        記錄每步的結果
        
        參數：
        -----
        pl : float - 損益
        sharpe : float - 夏普比率
        penalty : float - 懲罰項
        reward : float - 最終獎勵
        """
        df = pd.DataFrame({
            'e': self.episode,
            'bar': self.bar,
            'xt': self.xt,
            'yt': self.yt,
            'pv': self.portfolio_value,
            'pv_new': self.portfolio_value_new,
            'p&l[$]': pl,
            'p&l[%]': pl / self.portfolio_value_new * 100 if self.portfolio_value_new > 0 else 0,
            'sharpe': sharpe,
            'penalty': penalty,
            'reward': reward,
            'Xt': self.state[0],
            'Yt': self.state[1],
            'Xt_new': self.new_state[0],
            'Yt_new': self.new_state[1],
            'r': self.short_rate,
            'mu': self.index_drift,
            'sigma': self.volatility
        }, index=[0])
        
        self.portfolios = pd.concat((self.portfolios, df), ignore_index=True)
    
    def step(self, action):
        """
        執行一步動作
        
        參數：
        -----
        action : float - 風險資產配置比例 [0, 1]
        
        返回：
        -----
        tuple : (new_state, reward, done, truncated, info)
        """
        self.bar += 1
        self.new_state, info = self._get_state()
        
        if self.bar == 1:
            # 初始配置
            self.xt = action
            self.yt = 1 - action
            pl = 0.0
            sharpe = 0.0
            penalty = 0.0
            reward = 0.0
            self.add_results(pl, sharpe, penalty, reward)
        else:
            # 計算投資組合新價值
            self.portfolio_value_new = (
                self.xt * self.portfolio_value * self.new_state[0] / self.state[0] +
                self.yt * self.portfolio_value * self.new_state[1] / self.state[1]
            )
            
            # 計算損益
            pl = self.portfolio_value_new - self.portfolio_value
            
            # 計算單期報酬率
            period_return = pl / self.portfolio_value if self.portfolio_value > 0 else 0
            self.returns_history.append(period_return)
            
            # 計算配置調整懲罰項
            penalty = self._calculate_penalty(self.xt, action)
            
            # 計算滾動夏普比率
            sharpe = self._calculate_rolling_sharpe()
            
            # ========================================
            # 核心獎勵函數：夏普比率 - 懲罰項
            # ========================================
            reward = sharpe - penalty
            
            # 更新配置
            self.xt = action
            self.yt = 1 - action
            
            # 記錄結果
            self.add_results(pl, sharpe, penalty, reward)
            
            # 更新投資組合價值
            self.portfolio_value = self.portfolio_value_new
        
        # 檢查是否結束
        done = self.bar == len(self.data) - 1
        
        self.state = self.new_state
        
        return self.state, float(reward), done, False, info


# ============================================================================
# 第三部分：DQL 代理類別
# ============================================================================

class DQLInvestingAgent:
    """
    Deep Q-Learning 投資代理
    
    使用神經網路來學習最優的資產配置策略
    """
    
    def __init__(self, env, hidden_units=128, learning_rate=0.001,
                 gamma=0.95, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, batch_size=32, memory_size=2000):
        """
        初始化 DQL 代理
        
        參數：
        -----
        env : InvestingEnvironment - 投資環境
        hidden_units : int - 隱藏層神經元數量
        learning_rate : float - 學習率
        gamma : float - 折扣因子
        epsilon : float - 探索率初始值
        epsilon_min : float - 探索率最小值
        epsilon_decay : float - 探索率衰減係數
        batch_size : int - 批次大小
        memory_size : int - 經驗回放記憶體大小
        """
        self.env = env
        self.n_features = env.osn
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # 經驗回放記憶體
        self.memory = deque(maxlen=memory_size)
        
        # 配置位置索引（用於最優動作計算）
        self.xp = 3  # xt 在狀態向量中的位置
        self.yp = 4  # yt 在狀態向量中的位置
        
        # 建立神經網路模型
        self._create_model()
    
    def _create_model(self):
        """建立深度神經網路模型"""
        if HAS_TF:
            self.model = Sequential([
                Dense(self.hidden_units, input_dim=self.n_features, activation='relu'),
                Dense(self.hidden_units, activation='relu'),
                Dense(1, activation='linear')  # 輸出預測的總獎勵
            ])
            self.model.compile(
                loss='mse',
                optimizer=Adam(learning_rate=self.learning_rate)
            )
        else:
            # 簡化版本：使用線性模型
            self.weights = np.random.randn(self.n_features) * 0.01
            self.bias = 0.0
    
    def _predict(self, state):
        """預測給定狀態的價值"""
        if HAS_TF:
            return self.model.predict(state, verbose=0)[0, 0]
        else:
            return np.dot(state.flatten(), self.weights) + self.bias
    
    def _fit(self, state, target):
        """訓練模型"""
        if HAS_TF:
            self.model.fit(state, np.array([target]), epochs=1, verbose=0)
        else:
            # 簡化版本：梯度下降
            pred = self._predict(state)
            error = target - pred
            self.weights += self.learning_rate * error * state.flatten()
            self.bias += self.learning_rate * error
    
    def _reshape(self, state):
        """重塑狀態為模型輸入格式"""
        return np.reshape(state, [1, self.n_features])
    
    def opt_action(self, state):
        """
        計算最優動作
        
        透過最佳化找到使預測獎勵最大化的動作
        
        參數：
        -----
        state : array - 當前狀態
        
        返回：
        -----
        float : 最優動作（風險資產配置比例）
        """
        bounds = [(0, 1)]  # 動作範圍
        
        def objective(x):
            """目標函數：最大化預測獎勵"""
            s = state.copy()
            s[0, self.xp] = x[0]  # 更新風險資產配置
            s[0, self.yp] = 1 - x[0]  # 更新無風險資產配置
            return -self._predict(s)  # 負號因為 minimize
        
        try:
            result = minimize(
                objective,
                x0=[0.5],  # 初始猜測
                bounds=bounds,
                method='L-BFGS-B'
            )
            action = result.x[0]
        except:
            action = self.env.action_space.sample()
        
        return np.clip(action, 0, 1)
    
    def act(self, state):
        """
        選擇動作（ε-貪婪策略）
        
        參數：
        -----
        state : array - 當前狀態
        
        返回：
        -----
        float : 選擇的動作
        """
        if random.random() <= self.epsilon:
            return self.env.action_space.sample()
        
        state = self._reshape(state)
        return self.opt_action(state)
    
    def remember(self, state, action, reward, next_state, done):
        """儲存經驗到記憶體"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """
        經驗回放訓練
        
        從記憶體中隨機抽取批次進行訓練
        """
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            state = self._reshape(state)
            next_state = self._reshape(next_state)
            
            target = reward
            
            if not done:
                # 計算下一狀態的最優動作
                next_action = self.opt_action(next_state)
                ns = next_state.copy()
                ns[0, self.xp] = next_action
                ns[0, self.yp] = 1 - next_action
                
                # TD 目標
                target += self.gamma * self._predict(ns)
            
            # 更新當前狀態-動作的價值
            state[0, self.xp] = action
            state[0, self.yp] = 1 - action
            self._fit(state, target)
        
        # 衰減探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def learn(self, episodes, verbose=True):
        """
        訓練代理
        
        參數：
        -----
        episodes : int - 訓練回合數
        verbose : bool - 是否顯示訓練進度
        """
        rewards_history = []
        
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            total_reward = 0
            
            for _ in range(self.env.steps):
                action = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                
                self.remember(state, action, reward, next_state, done)
                self.replay()
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            rewards_history.append(total_reward)
            
            if verbose and e % 10 == 0:
                avg_reward = np.mean(rewards_history[-10:])
                print(f"Episode {e:4d} | "
                      f"Total Reward: {total_reward:8.2f} | "
                      f"Avg(10): {avg_reward:8.2f} | "
                      f"Epsilon: {self.epsilon:.4f}")
        
        return rewards_history
    
    def test(self, episodes, verbose=True):
        """
        測試代理
        
        參數：
        -----
        episodes : int - 測試回合數
        verbose : bool - 是否顯示測試結果
        """
        original_epsilon = self.epsilon
        self.epsilon = 0  # 測試時不探索
        
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            total_reward = 0
            
            for _ in range(self.env.steps):
                state_reshaped = self._reshape(state)
                action = self.opt_action(state_reshaped)
                next_state, reward, done, _, _ = self.env.step(action)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            if verbose:
                final_value = self.env.portfolio_value
                print(f"Test Episode {e} | "
                      f"Total Reward: {total_reward:.2f} | "
                      f"Final Value: {final_value:.4f}")
        
        self.epsilon = original_epsilon


# ============================================================================
# 第四部分：視覺化函數
# ============================================================================

def plot_training_results(rewards_history, window=10):
    """繪製訓練獎勵曲線"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(rewards_history, alpha=0.3, label='Episode Reward')
    
    # 移動平均
    if len(rewards_history) >= window:
        ma = pd.Series(rewards_history).rolling(window).mean()
        ax.plot(ma, linewidth=2, label=f'Moving Average ({window})')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_portfolio_performance(env, episode_num=None):
    """繪製投資組合表現"""
    if episode_num is not None:
        data = env.portfolios[env.portfolios['e'] == episode_num]
    else:
        # 使用最後一個 episode
        episode_num = env.portfolios['e'].max()
        data = env.portfolios[env.portfolios['e'] == episode_num]
    
    if len(data) == 0:
        print("No data available for plotting")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 圖1：資產價格
    ax1 = axes[0, 0]
    ax1.plot(data['bar'], data['Xt'], 'g--', label='Risky Asset (Xt)', linewidth=1.5)
    ax1.plot(data['bar'], data['Yt'], 'b:', label='Risk-free Asset (Yt)', linewidth=1.5)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Price')
    ax1.set_title('Asset Prices')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 圖2：投資組合價值
    ax2 = axes[0, 1]
    ax2.plot(data['bar'], data['pv_new'], 'r-', label='Portfolio Value', linewidth=2)
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Initial Value')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Value')
    ax2.set_title('Portfolio Value Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 圖3：資產配置
    ax3 = axes[1, 0]
    ax3.plot(data['bar'], data['xt'], 'g-', label='Risky Allocation (xt)', linewidth=1.5)
    ax3.plot(data['bar'], data['yt'], 'b--', label='Risk-free Allocation (yt)', linewidth=1.5)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Allocation')
    ax3.set_title('Asset Allocation Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)
    
    # 圖4：獎勵分解
    ax4 = axes[1, 1]
    ax4.plot(data['bar'], data['sharpe'], 'b-', label='Rolling Sharpe', linewidth=1.5)
    ax4.plot(data['bar'], data['penalty'], 'r--', label='Penalty', linewidth=1.5)
    ax4.plot(data['bar'], data['reward'], 'g-', label='Reward (Sharpe - Penalty)', 
             linewidth=2, alpha=0.7)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Value')
    ax4.set_title('Reward Decomposition')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Episode {episode_num} Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_reward_components_distribution(env):
    """繪製獎勵組成分布"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 夏普比率分布
    axes[0].hist(env.portfolios['sharpe'].dropna(), bins=30, color='blue', 
                 edgecolor='black', alpha=0.7)
    axes[0].axvline(x=env.portfolios['sharpe'].mean(), color='r', 
                    linestyle='--', label=f"Mean: {env.portfolios['sharpe'].mean():.3f}")
    axes[0].set_xlabel('Rolling Sharpe Ratio')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Sharpe Ratio Distribution')
    axes[0].legend()
    
    # 懲罰項分布
    axes[1].hist(env.portfolios['penalty'].dropna(), bins=30, color='red', 
                 edgecolor='black', alpha=0.7)
    axes[1].axvline(x=env.portfolios['penalty'].mean(), color='b', 
                    linestyle='--', label=f"Mean: {env.portfolios['penalty'].mean():.3f}")
    axes[1].set_xlabel('Rebalancing Penalty')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Penalty Distribution')
    axes[1].legend()
    
    # 總獎勵分布
    axes[2].hist(env.portfolios['reward'].dropna(), bins=30, color='green', 
                 edgecolor='black', alpha=0.7)
    axes[2].axvline(x=env.portfolios['reward'].mean(), color='r', 
                    linestyle='--', label=f"Mean: {env.portfolios['reward'].mean():.3f}")
    axes[2].set_xlabel('Total Reward')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Total Reward Distribution')
    axes[2].legend()
    
    plt.tight_layout()
    return fig


def analyze_allocation_stability(env):
    """分析配置穩定性"""
    # 計算配置變化
    allocation_changes = env.portfolios.groupby('e')['xt'].apply(
        lambda x: np.abs(x.diff()).mean()
    )
    
    print("\n配置穩定性分析：")
    print(f"  平均配置變化: {allocation_changes.mean():.4f}")
    print(f"  最大配置變化: {allocation_changes.max():.4f}")
    print(f"  最小配置變化: {allocation_changes.min():.4f}")
    
    # 按 episode 統計
    episode_stats = env.portfolios.groupby('e').agg({
        'xt': 'mean',
        'pv_new': 'last',
        'sharpe': 'mean',
        'penalty': 'mean',
        'reward': 'sum'
    }).round(4)
    
    print("\n各 Episode 統計：")
    print(episode_stats.tail(10))
    
    return episode_stats


# ============================================================================
# 第五部分：主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("強化學習投資策略 - 滾動夏普比率 + 配置調整懲罰項")
    print("=" * 70)
    
    # 設定隨機種子
    random.seed(42)
    np.random.seed(42)
    if HAS_TF:
        tf.random.set_seed(42)
    
    # 創建投資環境
    print("\n創建投資環境...")
    env = InvestingEnvironment(
        S0=1.0,
        T=1.0,
        r_=[0.0, 0.025, 0.05],        # 無風險利率候選
        mu_=[0.05, 0.1, 0.15],        # 漂移率候選
        sigma_=[0.1, 0.2, 0.3],       # 波動率候選
        steps=252,                     # 一年交易日
        amount=1.0,                    # 初始投資
        rolling_window=20,             # 20日滾動窗口
        penalty_factor=1.0             # 懲罰係數
    )
    
    print(f"  狀態空間維度: {env.osn}")
    print(f"  動作空間: 連續 [0, 1]")
    print(f"  滾動窗口: {env.rolling_window} 天")
    print(f"  懲罰係數: {env.penalty_factor}")
    
    # 創建 DQL 代理
    print("\n創建 DQL 代理...")
    agent = DQLInvestingAgent(
        env=env,
        hidden_units=64,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=32
    )
    
    # 訓練代理
    print("\n開始訓練...")
    episodes = 50
    rewards_history = agent.learn(episodes, verbose=True)
    
    print(f"\n訓練完成！")
    print(f"  最終探索率: {agent.epsilon:.4f}")
    print(f"  平均獎勵 (最後10回合): {np.mean(rewards_history[-10:]):.2f}")
    
    # 繪製訓練結果
    print("\n繪製訓練結果...")
    fig1 = plot_training_results(rewards_history)
    fig1.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    print("  已保存: training_progress.png")
    
    # 測試代理
    print("\n測試代理...")
    env.portfolios = pd.DataFrame()  # 重置記錄
    agent.test(5, verbose=True)
    
    # 繪製投資組合表現
    print("\n繪製投資組合表現...")
    fig2 = plot_portfolio_performance(env)
    if fig2:
        fig2.savefig('portfolio_performance.png', dpi=150, bbox_inches='tight')
        print("  已保存: portfolio_performance.png")
    
    # 繪製獎勵組成分布
    print("\n繪製獎勵組成分布...")
    fig3 = plot_reward_components_distribution(env)
    fig3.savefig('reward_components.png', dpi=150, bbox_inches='tight')
    print("  已保存: reward_components.png")
    
    # 分析配置穩定性
    stats = analyze_allocation_stability(env)
    
    plt.close('all')
    
    print("\n" + "=" * 70)
    print("程式執行完成！")
    print("=" * 70)
    
    print("""
    
關鍵要點總結：
=============

1. 獎勵函數設計：
   reward = 滾動夏普比率 - 配置調整懲罰項
   
   - 滾動夏普比率：衡量風險調整後的報酬
   - 配置調整懲罰項：(old_xt - new_xt)² × penalty_factor

2. 為什麼這樣設計：
   - 夏普比率鼓勵高風險調整報酬
   - 懲罰項避免過度交易（交易成本）
   - 結合兩者達到風險管理與成本控制的平衡

3. 滾動計算的優點：
   - 每步都有可計算的獎勵信號
   - 避免等到 episode 結束才有獎勵
   - 更符合馬可夫性質

4. 懲罰係數的影響：
   - 懲罰係數越大 → 配置越穩定，但可能錯失機會
   - 懲罰係數越小 → 配置更靈活，但交易成本增加
    """)