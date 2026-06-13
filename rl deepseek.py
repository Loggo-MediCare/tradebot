"""
改進版強化學習投資環境 - 動態市場適應 + 風險感知獎勵
===================================================

改進重點：
1. 增加市場狀態分類（牛市/熊市/震盪市）
2. 動態調整懲罰係數，避免過度交易但又不躺平
3. 加入風險感知獎勵，讓AI學會避險
4. 增加更多技術指標作為狀態輸入

作者：Claude AI - 改進版
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
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("警告：TensorFlow 未安裝，將使用簡化版本")


# ============================================================================
# 第一部分：輔助類別定義
# ============================================================================

class ObservationSpace:
    """改進版觀察空間 - 增加技術指標"""
    def __init__(self, n):
        self.shape = (n,)


class ActionSpace:
    """
    動作空間類別
    
    動作為連續值，範圍 [0, 1]
    0: 全現金避險
    1: 全倉位做多
    """
    def __init__(self, n):
        self.n = n
    
    def seed(self, seed):
        random.seed(seed)
    
    def sample(self):
        """隨機抽取一個動作（0到1之間的浮點數）"""
        return random.random()


# ============================================================================
# 第二部分：改進版投資環境類別
# ============================================================================

class EnhancedInvestingEnvironment:
    """
    改進版投資環境 - 動態市場適應
    
    主要改進：
    1. 市場狀態檢測（牛/熊/震盪）
    2. 動態懲罰係數
    3. 風險感知獎勵
    4. 更多狀態特徵
    
    狀態空間（15維）：
    - Xt: 風險資產當前價格（標準化）
    - Yt: 無風險資產當前價格（標準化）
    - Returns: 過去5期報酬率
    - Volatility: 過去20期波動率
    - Trend: 價格趨勢（1=牛市, -1=熊市, 0=震盪）
    - xt: 當前風險資產配置比例
    - yt: 當前無風險資產配置比例
    - Market_stress: 市場壓力指標
    """
    
    def __init__(self, S0=1.0, T=1.0, 
                 steps=252, amount=1.0, 
                 rolling_window=20, base_penalty=0.5,
                 market_diversity=True):
        """
        初始化改進版投資環境
        
        參數：
        -----
        S0 : float - 初始資產價格
        T : float - 投資期間（年）
        steps : int - 總步數（交易日數）
        amount : float - 初始投資金額
        rolling_window : int - 計算技術指標的窗口大小
        base_penalty : float - 基礎懲罰係數
        market_diversity : bool - 是否強制市場多樣性
        """
        # 環境設定
        self.initial_value = S0
        self.maturity = T
        self.steps = steps
        self.initial_balance = amount
        self.rolling_window = rolling_window
        self.base_penalty = base_penalty
        self.market_diversity = market_diversity
        
        # 市場參數範圍（確保多樣性）
        if market_diversity:
            # 強制包含熊市場景
            self.market_modes = ['bull', 'bear', 'sideways']
            self.mode_probs = [0.4, 0.4, 0.2]  # 增加熊市機率
        else:
            self.market_modes = ['bull', 'bear', 'sideways']
            self.mode_probs = [0.6, 0.2, 0.2]
        
        # 空間定義
        self.observation_space = ObservationSpace(15)  # 增加特徵維度
        self.osn = self.observation_space.shape[0]
        self.action_space = ActionSpace(1)
        
        # 技術指標歷史
        self.price_history = deque(maxlen=50)
        self.return_history = deque(maxlen=50)
        
        # 初始化
        self._reset_tracking()
        self.portfolios = pd.DataFrame()
        self.episode = 0
        
        # 動態懲罰參數
        self.penalty_multiplier = 1.0
        self.volatility_threshold = 0.15
        
    def _reset_tracking(self):
        """重置追蹤變數"""
        self.market_mode = None
        self.current_trend = 0
        self.market_stress = 0.0
        self.dynamic_penalty = self.base_penalty
    
    def _generate_market_data(self):
        """
        生成具有多樣性的市場數據
        
        確保訓練中包含熊市場景
        """
        # 隨機選擇市場模式
        self.market_mode = random.choices(
            self.market_modes, 
            weights=self.mode_probs, 
            k=1
        )[0]
        
        # 根據市場模式設定參數
        if self.market_mode == 'bull':
            # 牛市：正漂移，中等波動
            self.index_drift = random.uniform(0.08, 0.20)
            self.volatility = random.uniform(0.15, 0.25)
            self.short_rate = random.uniform(0.02, 0.05)
            
        elif self.market_mode == 'bear':
            # 熊市：負漂移，高波動
            self.index_drift = random.uniform(-0.15, -0.05)
            self.volatility = random.uniform(0.25, 0.40)
            self.short_rate = random.uniform(0.00, 0.03)
            
        else:  # sideways
            # 震盪市：接近零漂移，低波動
            self.index_drift = random.uniform(-0.02, 0.02)
            self.volatility = random.uniform(0.10, 0.20)
            self.short_rate = random.uniform(0.03, 0.06)
        
        self.dt = self.maturity / self.steps
        
        # 生成價格路徑
        self._generate_price_path()
        
        # 計算技術指標
        self._calculate_technical_indicators()
    
    def _generate_price_path(self):
        """生成價格路徑（加入均值回歸特性）"""
        s = [self.initial_value]
        
        # 隨機漫步參數
        alpha = 0.3  # 均值回歸強度
        theta = self.initial_value  # 長期均值
        
        for t in range(1, self.steps + 1):
            # 均值回歸項
            mean_reversion = alpha * (theta - s[t-1]) * self.dt
            
            # 隨機項
            drift = self.index_drift * self.dt
            diffusion = self.volatility * math.sqrt(self.dt) * random.gauss(0, 1)
            
            # 新價格
            st = s[t-1] * math.exp(mean_reversion + drift + diffusion)
            s.append(st)
        
        # 建立數據框
        self.data = pd.DataFrame(s, columns=['Xt'])
        
        # 計算無風險資產價值
        self.data['Yt'] = self.initial_value * np.exp(
            self.short_rate * np.arange(len(self.data)) * self.dt
        )
        
        # 計算報酬率
        self.data['Returns'] = self.data['Xt'].pct_change()
        
        # 初始化歷史數據
        self.price_history.clear()
        self.return_history.clear()
        for price, ret in zip(self.data['Xt'], self.data['Returns']):
            self.price_history.append(price)
            if not np.isnan(ret):
                self.return_history.append(ret)
    
    def _calculate_technical_indicators(self):
        """計算技術指標"""
        # 移動平均線
        self.data['MA20'] = self.data['Xt'].rolling(window=20).mean()
        self.data['MA50'] = self.data['Xt'].rolling(window=50).mean()
        
        # 波動率
        self.data['Volatility'] = self.data['Returns'].rolling(window=20).std()
        
        # RSI
        delta = self.data['Xt'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = self.data['Xt'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['Xt'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['Signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        
        # 填充NaN值
        self.data = self.data.fillna(method='bfill').fillna(method='ffill')
    
    def _detect_market_trend(self):
        """檢測市場趨勢"""
        if len(self.price_history) < 20:
            return 0
        
        # 使用移動平均線判斷趨勢
        current_price = self.price_history[-1]
        ma20 = np.mean(list(self.price_history)[-20:])
        ma50 = np.mean(list(self.price_history)[-50:]) if len(self.price_history) >= 50 else ma20
        
        # 趨勢判斷
        if current_price > ma20 > ma50:
            return 1  # 牛市
        elif current_price < ma20 < ma50:
            return -1  # 熊市
        else:
            return 0  # 震盪市
    
    def _calculate_market_stress(self):
        """計算市場壓力指標"""
        if len(self.return_history) < 10:
            return 0.0
        
        returns = np.array(list(self.return_history)[-10:])
        
        # 波動率壓力
        volatility = np.std(returns) if len(returns) > 1 else 0.01
        
        # 下跌壓力（負偏度）
        negative_returns = returns[returns < 0]
        downside_pressure = len(negative_returns) / len(returns) if len(returns) > 0 else 0
        
        # 極端值壓力
        var_95 = np.percentile(returns, 5) if len(returns) > 5 else -0.01
        
        # 綜合壓力指標
        stress = (
            0.4 * min(volatility / 0.02, 1.0) +  # 波動率貢獻
            0.3 * downside_pressure +             # 下跌天數貢獻
            0.3 * min(abs(var_95) / 0.03, 1.0)    # 風險價值貢獻
        )
        
        return min(stress, 1.0)
    
    def _get_state(self):
        """
        獲取改進版狀態
        
        返回：
        -----
        tuple : (state_array, info_dict)
        """
        if self.bar < 20:  # 確保有足夠數據
            # 返回簡化狀態
            state = np.zeros(self.osn)
            state[0] = self.data['Xt'].iloc[self.bar]
            state[1] = self.data['Yt'].iloc[self.bar]
            state[-2] = self.xt
            state[-1] = self.yt
            return state, {}
        
        # 基本價格
        Xt = self.data['Xt'].iloc[self.bar]
        Yt = self.data['Yt'].iloc[self.bar]
        
        # 技術指標
        ma20 = self.data['MA20'].iloc[self.bar]
        ma50 = self.data['MA50'].iloc[self.bar] if self.bar >= 50 else ma20
        volatility = self.data['Volatility'].iloc[self.bar]
        rsi = self.data['RSI'].iloc[self.bar]
        macd = self.data['MACD'].iloc[self.bar]
        signal = self.data['Signal'].iloc[self.bar]
        
        # 歷史報酬率（最近5期）
        recent_returns = []
        for i in range(1, 6):
            if self.bar - i >= 0:
                recent_returns.append(self.data['Returns'].iloc[self.bar - i])
            else:
                recent_returns.append(0.0)
        
        # 檢測趨勢
        self.current_trend = self._detect_market_trend()
        
        # 計算市場壓力
        self.market_stress = self._calculate_market_stress()
        
        # 動態調整懲罰係數
        self._adjust_dynamic_penalty()
        
        # 構建狀態向量
        state = np.array([
            Xt, Yt,                          # 當前價格
            Xt - Yt,                         # 價格差異
            ma20, ma50,                      # 移動平均線
            volatility,                      # 波動率
            rsi / 100,                       # RSI（正規化）
            macd, signal,                    # MACD
            *recent_returns,                 # 歷史報酬率
            self.current_trend,              # 趨勢
            self.market_stress,              # 市場壓力
            self.xt, self.yt                 # 當前配置
        ])
        
        info = {
            'market_mode': self.market_mode,
            'trend': self.current_trend,
            'stress': self.market_stress,
            'dynamic_penalty': self.dynamic_penalty
        }
        
        return state, info
    
    def _adjust_dynamic_penalty(self):
        """動態調整懲罰係數"""
        # 在高波動或高壓力市場降低懲罰，鼓勵避險
        if self.market_stress > 0.7 or self.data['Volatility'].iloc[self.bar] > self.volatility_threshold:
            self.dynamic_penalty = self.base_penalty * 0.5  # 降低懲罰
        elif self.market_stress < 0.3 and self.data['Volatility'].iloc[self.bar] < 0.1:
            self.dynamic_penalty = self.base_penalty * 1.5  # 增加懲罰
        else:
            self.dynamic_penalty = self.base_penalty
    
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
        self._reset_tracking()
        self._generate_market_data()
        
        # 重置收益記錄
        self.returns_history = []
        self.actions_history = []
        
        self.state, info = self._get_state()
        return self.state, info
    
    def _calculate_enhanced_sharpe(self):
        """
        計算增強版夏普比率
        
        考慮市場狀態的風險調整
        """
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array(self.returns_history)
        
        # 使用最近 rolling_window 期的數據
        if len(returns) > self.rolling_window:
            returns = returns[-self.rolling_window:]
        
        # 計算平均報酬率
        mean_return = np.mean(returns)
        
        # 計算下行風險（Downside Risk）
        negative_returns = returns[returns < 0]
        downside_risk = np.std(negative_returns) if len(negative_returns) > 1 else 0.001
        
        # 在熊市中更關注下行風險
        if self.current_trend == -1:  # 熊市
            risk_measure = downside_risk * 2.0  # 加倍懲罰下行風險
        elif self.current_trend == 1:  # 牛市
            risk_measure = np.std(returns) if len(returns) > 1 else 0.001
        else:  # 震盪市
            risk_measure = np.std(returns) if len(returns) > 1 else 0.001
        
        # 避免除以零
        if risk_measure < 0.001:
            risk_measure = 0.001
        
        # 增強版夏普比率（考慮市場狀態）
        enhanced_sharpe = mean_return / risk_measure
        
        # 在熊市中，正報酬給予額外獎勵
        if self.current_trend == -1 and mean_return > 0:
            enhanced_sharpe *= 1.5
        
        # 裁剪到合理範圍
        enhanced_sharpe = np.clip(enhanced_sharpe, -5, 5)
        
        return enhanced_sharpe
    
    def _calculate_adaptive_penalty(self, old_allocation, new_allocation):
        """
        計算自適應懲罰項
        
        懲罰根據市場狀態動態調整
        """
        base_change = abs(old_allocation - new_allocation)
        
        # 市場壓力高時降低懲罰，鼓勵調整
        if self.market_stress > 0.7:
            penalty_multiplier = 0.5
        # 市場平穩時正常懲罰
        elif self.market_stress < 0.3:
            penalty_multiplier = 1.5
        else:
            penalty_multiplier = 1.0
        
        # 趨勢變化時的特殊處理
        if self.current_trend == -1 and new_allocation < 0.3:
            # 熊市中降低避險懲罰
            penalty_multiplier *= 0.7
        
        penalty = self.dynamic_penalty * penalty_multiplier * (base_change ** 2)
        
        # 避免極端懲罰
        return min(penalty, 1.0)
    
    def _calculate_trend_reward(self):
        """
        計算趨勢跟隨獎勵
        
        鼓勵在牛市中做多，熊市中避險
        """
        if self.current_trend == 1:  # 牛市
            # 鼓勵增加風險暴露
            trend_reward = min(self.xt * 0.1, 0.05)
        elif self.current_trend == -1:  # 熊市
            # 鼓勵減少風險暴露
            trend_reward = min((1 - self.xt) * 0.1, 0.05)
        else:  # 震盪市
            trend_reward = 0.0
        
        return trend_reward
    
    def add_results(self, pl, sharpe, penalty, trend_reward, total_reward):
        """
        記錄每步的結果
        """
        df = pd.DataFrame({
            'e': self.episode,
            'bar': self.bar,
            'market_mode': self.market_mode,
            'trend': self.current_trend,
            'stress': self.market_stress,
            'xt': self.xt,
            'yt': self.yt,
            'pv': self.portfolio_value,
            'pv_new': self.portfolio_value_new,
            'p&l[$]': pl,
            'p&l[%]': pl / self.portfolio_value_new * 100 if self.portfolio_value_new > 0 else 0,
            'sharpe': sharpe,
            'penalty': penalty,
            'trend_reward': trend_reward,
            'total_reward': total_reward,
            'Xt': self.state[0],
            'Yt': self.state[1],
            'volatility': self.data['Volatility'].iloc[self.bar] if self.bar < len(self.data) else 0,
            'rsi': self.data['RSI'].iloc[self.bar] if self.bar < len(self.data) else 50,
            'dynamic_penalty': self.dynamic_penalty,
            'mu': self.index_drift,
            'sigma': self.volatility
        }, index=[0])
        
        self.portfolios = pd.concat((self.portfolios, df), ignore_index=True)
    
    def step(self, action):
        """
        執行一步動作
        """
        self.bar += 1
        
        # 記錄動作歷史
        self.actions_history.append(action)
        
        # 獲取新狀態
        self.new_state, info = self._get_state()
        
        if self.bar == 1:
            # 初始配置
            self.xt = action
            self.yt = 1 - action
            pl = 0.0
            sharpe = 0.0
            penalty = 0.0
            trend_reward = 0.0
            total_reward = 0.0
            self.add_results(pl, sharpe, penalty, trend_reward, total_reward)
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
            
            # 計算懲罰項
            penalty = self._calculate_adaptive_penalty(self.xt, action)
            
            # 計算增強版夏普比率
            sharpe = self._calculate_enhanced_sharpe()
            
            # 計算趨勢跟隨獎勵
            trend_reward = self._calculate_trend_reward()
            
            # ========================================
            # 改進版獎勵函數：多項組合
            # ========================================
            # 基礎：夏普比率
            base_reward = sharpe
            
            # 避險獎勵（熊市中降低倉位）
            if self.current_trend == -1 and action < 0.3:
                hedge_bonus = 0.2 * (1 - action)
            else:
                hedge_bonus = 0.0
            
            # 最終獎勵
            total_reward = base_reward - penalty + trend_reward + hedge_bonus
            
            # 更新配置
            self.xt = action
            self.yt = 1 - action
            
            # 記錄結果
            self.add_results(pl, sharpe, penalty, trend_reward, total_reward)
            
            # 更新投資組合價值
            self.portfolio_value = self.portfolio_value_new
        
        # 檢查是否結束
        done = self.bar == len(self.data) - 1
        
        self.state = self.new_state
        
        return self.state, float(total_reward), done, False, info


# ============================================================================
# 第三部分：改進版DQL代理類別
# ============================================================================

class EnhancedDQLAgent:
    """
    改進版 Deep Q-Learning 投資代理
    
    改進重點：
    1. 使用更複雜的神經網絡結構
    2. 加入風險感知層
    3. 實現經驗優先回放
    4. 定期更新目標網絡
    """
    
    def __init__(self, env, hidden_units=128, learning_rate=0.001,
                 gamma=0.95, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, batch_size=32, memory_size=2000,
                 target_update_freq=100, use_lstm=False):
        """
        初始化改進版 DQL 代理
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
        self.target_update_freq = target_update_freq
        self.use_lstm = use_lstm
        
        # 經驗回放記憶體（帶優先級）
        self.memory = deque(maxlen=memory_size)
        self.priorities = deque(maxlen=memory_size)
        
        # 配置位置索引
        self.xp = -2  # xt 在狀態向量中的位置
        self.yp = -1  # yt 在狀態向量中的位置
        
        # 計數器
        self.train_step = 0
        
        # 建立神經網路模型
        self._create_models()
    
    def _create_models(self):
        """建立改進版神經網路模型"""
        if HAS_TF:
            # 主網絡
            self.model = self._build_network()
            self.model.compile(
                loss='mse',
                optimizer=Adam(learning_rate=self.learning_rate)
            )
            
            # 目標網絡（用於穩定訓練）
            self.target_model = self._build_network()
            self.target_model.set_weights(self.model.get_weights())
        else:
            # 簡化版本
            self.weights = np.random.randn(self.n_features) * 0.01
            self.bias = 0.0
    
    def _build_network(self):
        """構建神經網絡結構"""
        model = Sequential()
        
        # 輸入層
        model.add(Dense(256, input_dim=self.n_features, activation='relu'))
        model.add(Dropout(0.3))
        
        # 隱藏層
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        
        # 風險感知層
        model.add(Dense(32, activation='relu'))
        
        # 輸出層
        model.add(Dense(1, activation='linear'))
        
        return model
    
    def _predict(self, state, use_target=False):
        """預測給定狀態的價值"""
        if HAS_TF:
            if use_target:
                return self.target_model.predict(state, verbose=0)[0, 0]
            else:
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
    
    def _calculate_action_confidence(self, state, action):
        """
        計算動作信心分數
        
        用於動態探索
        """
        state_reshaped = self._reshape(state)
        current_value = self._predict(state_reshaped)
        
        # 創建對比狀態
        alt_actions = [max(0, action - 0.1), min(1, action + 0.1)]
        alt_values = []
        
        for alt_action in alt_actions:
            alt_state = state_reshaped.copy()
            alt_state[0, self.xp] = alt_action
            alt_state[0, self.yp] = 1 - alt_action
            alt_values.append(self._predict(alt_state))
        
        # 信心分數：當前動作價值與替代動作的差異
        if alt_values:
            confidence = current_value - np.mean(alt_values)
            confidence = np.clip(confidence, -1, 1)
        else:
            confidence = 0.0
        
        return confidence
    
    def opt_action(self, state):
        """
        計算最優動作（改進版）
        
        加入風險約束
        """
        bounds = [(0, 1)]  # 動作範圍
        
        # 根據市場狀態調整邊界
        if hasattr(self.env, 'current_trend'):
            if self.env.current_trend == -1:  # 熊市
                # 限制最大風險暴露
                bounds = [(0, 0.5)]
            elif self.env.current_trend == 1:  # 牛市
                # 鼓勵更多風險暴露
                bounds = [(0.3, 1.0)]
        
        def objective(x):
            """目標函數：最大化預測獎勵"""
            s = state.copy()
            s[0, self.xp] = x[0]  # 更新風險資產配置
            s[0, self.yp] = 1 - x[0]  # 更新無風險資產配置
            
            # 加入風險懲罰
            value = self._predict(s)
            
            # 市場壓力高時懲罰高風險
            if hasattr(self.env, 'market_stress') and self.env.market_stress > 0.7:
                risk_penalty = x[0] * 0.5 * self.env.market_stress
                value -= risk_penalty
            
            return -value  # 負號因為 minimize
        
        try:
            result = minimize(
                objective,
                x0=[0.5],  # 初始猜測
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 50}
            )
            action = result.x[0]
        except:
            action = np.mean(bounds[0])
        
        return np.clip(action, bounds[0][0], bounds[0][1])
    
    def adaptive_epsilon(self):
        """
        自適應探索率
        
        根據訓練進度和市場狀態調整
        """
        # 基礎衰減
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 在市場不確定性高時增加探索
        if hasattr(self.env, 'market_stress') and self.env.market_stress > 0.7:
            self.epsilon = min(self.epsilon * 1.2, 0.5)
    
    def act(self, state):
        """
        選擇動作（改進版ε-貪婪策略）
        """
        # 計算信心分數
        confidence = self._calculate_action_confidence(state, 0.5)
        
        # 動態探索率
        effective_epsilon = self.epsilon * max(0.5, 1 - abs(confidence))
        
        if random.random() <= effective_epsilon:
            # 探索：根據市場狀態調整探索策略
            if hasattr(self.env, 'current_trend'):
                if self.env.current_trend == -1:  # 熊市
                    # 偏向保守探索
                    action = random.uniform(0, 0.4)
                elif self.env.current_trend == 1:  # 牛市
                    # 偏向積極探索
                    action = random.uniform(0.4, 1.0)
                else:  # 震盪市
                    action = random.random()
            else:
                action = random.random()
        else:
            # 利用
            state_reshaped = self._reshape(state)
            action = self.opt_action(state_reshaped)
        
        return np.clip(action, 0, 1)
    
    def remember(self, state, action, reward, next_state, done):
        """儲存經驗到記憶體（帶優先級）"""
        self.memory.append((state, action, reward, next_state, done))
        
        # 計算TD誤差作為優先級
        state_reshaped = self._reshape(state)
        next_state_reshaped = self._reshape(next_state)
        
        current_q = self._predict(state_reshaped)
        
        if done:
            target_q = reward
        else:
            next_action = self.opt_action(next_state_reshaped)
            ns = next_state_reshaped.copy()
            ns[0, self.xp] = next_action
            ns[0, self.yp] = 1 - next_action
            target_q = reward + self.gamma * self._predict(ns, use_target=True)
        
        td_error = abs(target_q - current_q)
        self.priorities.append(td_error)
    
    def replay(self):
        """
        改進版經驗回放訓練
        
        使用優先級抽樣
        """
        if len(self.memory) < self.batch_size:
            return
        
        # 優先級抽樣
        priorities = np.array(self.priorities) + 1e-6  # 避免零優先級
        probabilities = priorities / np.sum(priorities)
        
        indices = np.random.choice(
            len(self.memory), 
            size=self.batch_size, 
            p=probabilities
        )
        
        batch = [self.memory[i] for i in indices]
        
        for state, action, reward, next_state, done in batch:
            state = self._reshape(state)
            next_state = self._reshape(next_state)
            
            target = reward
            
            if not done:
                next_action = self.opt_action(next_state)
                ns = next_state.copy()
                ns[0, self.xp] = next_action
                ns[0, self.yp] = 1 - next_action
                target += self.gamma * self._predict(ns, use_target=True)
            
            # 更新當前狀態-動作的價值
            state[0, self.xp] = action
            state[0, self.yp] = 1 - action
            self._fit(state, target)
        
        # 更新探索率
        self.adaptive_epsilon()
        
        # 定期更新目標網絡
        self.train_step += 1
        if HAS_TF and self.train_step % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
    
    def learn(self, episodes, verbose=True):
        """
        訓練代理
        """
        rewards_history = []
        market_modes = []
        
        for e in range(1, episodes + 1):
            state, info = self.env.reset()
            total_reward = 0
            
            # 記錄市場模式
            if 'market_mode' in info:
                market_modes.append(info['market_mode'])
            
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
                
                # 分析市場模式分布
                recent_modes = market_modes[-10:] if len(market_modes) >= 10 else market_modes
                if recent_modes:
                    bull_count = recent_modes.count('bull')
                    bear_count = recent_modes.count('bear')
                    side_count = recent_modes.count('sideways')
                else:
                    bull_count = bear_count = side_count = 0
                
                print(f"Episode {e:4d} | "
                      f"Reward: {total_reward:7.2f} | "
                      f"Avg(10): {avg_reward:7.2f} | "
                      f"Eps: {self.epsilon:.3f} | "
                      f"Market: BULL={bull_count} BEAR={bear_count} SIDE={side_count}")
        
        return rewards_history, market_modes
    
    def test(self, episodes=5, test_mode='bear', verbose=True):
        """
        改進版測試
        
        可以指定測試模式
        """
        original_epsilon = self.epsilon
        self.epsilon = 0  # 測試時不探索
        
        # 保存原始市場多樣性設置
        original_diversity = self.env.market_diversity
        self.env.market_diversity = False
        
        # 根據測試模式調整參數
        if test_mode == 'bear':
            # 強制熊市測試
            self.env.market_modes = ['bear']
            self.env.mode_probs = [1.0]
        elif test_mode == 'bull':
            # 強制牛市測試
            self.env.market_modes = ['bull']
            self.env.mode_probs = [1.0]
        else:
            # 混合測試
            self.env.market_modes = ['bull', 'bear', 'sideways']
            self.env.mode_probs = [0.33, 0.33, 0.34]
        
        test_results = []
        
        for e in range(1, episodes + 1):
            state, info = self.env.reset()
            total_reward = 0
            
            for _ in range(self.env.steps):
                state_reshaped = self._reshape(state)
                action = self.opt_action(state_reshaped)
                next_state, reward, done, _, _ = self.env.step(action)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            final_value = self.env.portfolio_value
            test_results.append({
                'episode': e,
                'total_reward': total_reward,
                'final_value': final_value,
                'market_mode': info.get('market_mode', 'unknown')
            })
            
            if verbose:
                print(f"Test {e} [{info.get('market_mode', 'unknown'):8s}] | "
                      f"Reward: {total_reward:7.2f} | "
                      f"Final: {final_value:.4f} | "
                      f"Avg Action: {np.mean(self.env.actions_history):.3f}")
        
        # 恢復原始設置
        self.epsilon = original_epsilon
        self.env.market_diversity = original_diversity
        
        return pd.DataFrame(test_results)


# ============================================================================
# 第四部分：改進版視覺化函數
# ============================================================================

def plot_enhanced_training_results(rewards_history, market_modes, window=10):
    """繪製改進版訓練結果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 圖1：訓練獎勵曲線
    ax1 = axes[0, 0]
    ax1.plot(rewards_history, alpha=0.3, label='Episode Reward', color='blue')
    
    # 移動平均
    if len(rewards_history) >= window:
        ma = pd.Series(rewards_history).rolling(window).mean()
        ax1.plot(ma, linewidth=2, label=f'Moving Average ({window})', color='red')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 圖2：市場模式分布
    ax2 = axes[0, 1]
    if market_modes:
        mode_counts = pd.Series(market_modes).value_counts()
        colors = {'bull': 'green', 'bear': 'red', 'sideways': 'gray'}
        mode_colors = [colors.get(mode, 'blue') for mode in mode_counts.index]
        
        bars = ax2.bar(range(len(mode_counts)), mode_counts.values, color=mode_colors)
        ax2.set_xticks(range(len(mode_counts)))
        ax2.set_xticklabels(mode_counts.index)
        ax2.set_ylabel('Count')
        ax2.set_title('Market Mode Distribution During Training')
        
        # 添加計數標籤
        for bar, count in zip(bars, mode_counts.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
    
    # 圖3：分市場模式的獎勵分布
    ax3 = axes[1, 0]
    if market_modes and len(market_modes) == len(rewards_history):
        df = pd.DataFrame({
            'mode': market_modes,
            'reward': rewards_history
        })
        
        modes = df['mode'].unique()
        reward_by_mode = []
        labels = []
        
        for mode in modes:
            mode_rewards = df[df['mode'] == mode]['reward']
            if len(mode_rewards) > 0:
                reward_by_mode.append(mode_rewards)
                labels.append(f'{mode} (n={len(mode_rewards)})')
        
        if reward_by_mode:
            box = ax3.boxplot(reward_by_mode, labels=labels, patch_artist=True)
            
            # 設置顏色
            colors = {'bull': 'lightgreen', 'bear': 'lightcoral', 'sideways': 'lightgray'}
            for i, label in enumerate(labels):
                for mode, color in colors.items():
                    if mode in label:
                        box['boxes'][i].set_facecolor(color)
                        break
            
            ax3.set_ylabel('Reward')
            ax3.set_title('Reward Distribution by Market Mode')
            ax3.grid(True, alpha=0.3)
    
    # 圖4：累積獎勵
    ax4 = axes[1, 1]
    cumulative_rewards = np.cumsum(rewards_history)
    ax4.plot(cumulative_rewards, linewidth=2, color='purple')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Cumulative Reward')
    ax4.set_title('Cumulative Training Reward')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_adaptive_portfolio_performance(env, episode_num=None):
    """繪製自適應投資組合表現"""
    if episode_num is not None:
        data = env.portfolios[env.portfolios['e'] == episode_num]
    else:
        episode_num = env.portfolios['e'].max()
        data = env.portfolios[env.portfolios['e'] == episode_num]
    
    if len(data) == 0:
        print("No data available for plotting")
        return None
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # 圖1：資產價格與趨勢
    ax1 = axes[0, 0]
    market_mode = data['market_mode'].iloc[0] if 'market_mode' in data.columns else 'unknown'
    
    # 根據市場模式設置顏色
    if market_mode == 'bull':
        price_color = 'green'
        trend_color = 'lightgreen'
    elif market_mode == 'bear':
        price_color = 'red'
        trend_color = 'lightcoral'
    else:
        price_color = 'gray'
        trend_color = 'lightgray'
    
    ax1.plot(data['bar'], data['Xt'], color=price_color, label='Risky Asset', linewidth=2)
    ax1.plot(data['bar'], data['Yt'], 'b--', label='Risk-free Asset', linewidth=1.5)
    
    # 標記趨勢
    if 'trend' in data.columns:
        bullish_periods = data[data['trend'] == 1]
        bearish_periods = data[data['trend'] == -1]
        
        if len(bullish_periods) > 0:
            ax1.scatter(bullish_periods['bar'], bullish_periods['Xt'], 
                       color='green', s=10, alpha=0.5, label='Bullish Signal')
        if len(bearish_periods) > 0:
            ax1.scatter(bearish_periods['bar'], bearish_periods['Xt'], 
                       color='red', s=10, alpha=0.5, label='Bearish Signal')
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Price')
    ax1.set_title(f'Asset Prices - Market Mode: {market_mode.upper()}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 圖2：投資組合價值
    ax2 = axes[0, 1]
    final_return = ((data['pv_new'].iloc[-1] / data['pv_new'].iloc[0]) - 1) * 100
    
    ax2.plot(data['bar'], data['pv_new'], 'k-', label='Portfolio Value', linewidth=2)
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Initial Value')
    
    # 填充績效區域
    ax2.fill_between(data['bar'], 1.0, data['pv_new'], 
                     where=(data['pv_new'] >= 1.0), 
                     alpha=0.3, color='green', label='Profit')
    ax2.fill_between(data['bar'], 1.0, data['pv_new'], 
                     where=(data['pv_new'] < 1.0), 
                     alpha=0.3, color='red', label='Loss')
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Value')
    ax2.set_title(f'Portfolio Value (Final Return: {final_return:.2f}%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 圖3：動態資產配置
    ax3 = axes[1, 0]
    ax3.plot(data['bar'], data['xt'], 'g-', label='Risky Allocation', linewidth=2)
    ax3.plot(data['bar'], data['yt'], 'b--', label='Risk-free Allocation', linewidth=1.5)
    
    # 添加市場壓力背景
    if 'stress' in data.columns:
        stress = data['stress'].fillna(0)
        ax3_twin = ax3.twinx()
        ax3_twin.fill_between(data['bar'], 0, stress, 
                              alpha=0.2, color='red', label='Market Stress')
        ax3_twin.set_ylabel('Market Stress', color='red')
        ax3_twin.set_ylim(0, 1)
        ax3_twin.tick_params(axis='y', labelcolor='red')
        
        # 合併圖例
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Allocation')
    ax3.set_title('Dynamic Asset Allocation with Market Stress')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)
    
    # 圖4：改進版獎勵分解
    ax4 = axes[1, 1]
    ax4.plot(data['bar'], data['sharpe'], 'b-', label='Enhanced Sharpe', linewidth=1.5)
    ax4.plot(data['bar'], data['penalty'], 'r--', label='Adaptive Penalty', linewidth=1.5)
    ax4.plot(data['bar'], data['trend_reward'], 'y:', label='Trend Reward', linewidth=1.5)
    ax4.plot(data['bar'], data['total_reward'], 'g-', label='Total Reward', linewidth=2, alpha=0.7)
    
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Value')
    ax4.set_title('Enhanced Reward Decomposition')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 圖5：技術指標
    ax5 = axes[2, 0]
    if 'volatility' in data.columns and 'rsi' in data.columns:
        # 波動率
        ax5_vol = ax5
        ax5_vol.plot(data['bar'], data['volatility'], 'b-', label='Volatility', linewidth=1.5)
        ax5_vol.set_xlabel('Time Step')
        ax5_vol.set_ylabel('Volatility', color='b')
        ax5_vol.tick_params(axis='y', labelcolor='b')
        
        # RSI
        ax5_rsi = ax5_vol.twinx()
        ax5_rsi.plot(data['bar'], data['rsi'], 'r--', label='RSI', linewidth=1)
        ax5_rsi.set_ylabel('RSI', color='r')
        ax5_rsi.tick_params(axis='y', labelcolor='r')
        ax5_rsi.axhline(y=70, color='r', linestyle=':', alpha=0.5)
        ax5_rsi.axhline(y=30, color='r', linestyle=':', alpha=0.5)
        ax5_rsi.set_ylim(0, 100)
        
        # 合併圖例
        lines1, labels1 = ax5_vol.get_legend_handles_labels()
        lines2, labels2 = ax5_rsi.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax5.set_title('Technical Indicators: Volatility & RSI')
        ax5.grid(True, alpha=0.3)
    
    # 圖6：動態懲罰係數
    ax6 = axes[2, 1]
    if 'dynamic_penalty' in data.columns:
        ax6.plot(data['bar'], data['dynamic_penalty'], 'purple', linewidth=2)
        ax6.set_xlabel('Time Step')
        ax6.set_ylabel('Penalty Factor')
        ax6.set_title('Dynamic Penalty Adjustment')
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Episode {episode_num} - Adaptive Portfolio Performance', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def analyze_adaptive_strategy(env, agent):
    """分析自適應策略表現"""
    print("\n" + "="*80)
    print("自適應策略分析報告")
    print("="*80)
    
    # 按市場模式分析
    if 'market_mode' in env.portfolios.columns:
        mode_stats = env.portfolios.groupby('market_mode').agg({
            'pv_new': 'last',
            'sharpe': 'mean',
            'total_reward': 'sum',
            'xt': 'mean'
        }).round(4)
        
        print("\n1. 各市場模式表現：")
        print(mode_stats)
        
        # 計算相對表現
        initial_value = env.initial_balance
        for mode in mode_stats.index:
            final_value = mode_stats.loc[mode, 'pv_new']
            returns = (final_value / initial_value - 1) * 100
            mode_stats.loc[mode, 'return_pct'] = returns
        
        print("\n2. 各模式報酬率：")
        print(mode_stats[['return_pct', 'sharpe', 'xt']])
    
    # 熊市表現分析
    bear_data = env.portfolios[env.portfolios['market_mode'] == 'bear']
    if len(bear_data) > 0:
        print("\n3. 熊市表現分析：")
        
        bear_episodes = bear_data['e'].unique()
        for ep in bear_episodes[:3]:  # 顯示前3個熊市episode
            ep_data = bear_data[bear_data['e'] == ep]
            if len(ep_data) > 0:
                start_val = ep_data['pv_new'].iloc[0]
                end_val = ep_data['pv_new'].iloc[-1]
                return_pct = (end_val / start_val - 1) * 100
                avg_allocation = ep_data['xt'].mean()
                
                print(f"   熊市 Episode {ep}: 報酬率={return_pct:6.2f}%, "
                      f"平均風險配置={avg_allocation:.3f}, "
                      f"最大下跌={min(ep_data['p&l[%]']):6.2f}%")
    
    # 策略穩定性分析
    allocation_changes = env.portfolios.groupby('e')['xt'].apply(
        lambda x: np.abs(x.diff()).mean()
    )
    
    print("\n4. 策略穩定性分析：")
    print(f"   平均配置變化: {allocation_changes.mean():.4f}")
    print(f"   最大配置變化: {allocation_changes.max():.4f}")
    print(f"   最小配置變化: {allocation_changes.min():.4f}")
    
    # 風險調整後表現
    if 'sharpe' in env.portfolios.columns:
        avg_sharpe = env.portfolios['sharpe'].mean()
        sharpe_std = env.portfolios['sharpe'].std()
        
        print("\n5. 風險調整後表現：")
        print(f"   平均夏普比率: {avg_sharpe:.4f}")
        print(f"   夏普比率波動: {sharpe_std:.4f}")
    
    return mode_stats if 'market_mode' in env.portfolios.columns else None


# ============================================================================
# 第五部分：主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("改進版強化學習投資策略 - 動態市場適應 + 風險感知")
    print("=" * 80)
    
    # 設定隨機種子
    random.seed(42)
    np.random.seed(42)
    if HAS_TF:
        tf.random.set_seed(42)
    
    # 創建改進版投資環境
    print("\n創建改進版投資環境...")
    env = EnhancedInvestingEnvironment(
        S0=1.0,
        T=1.0,
        steps=252,                     # 一年交易日
        amount=1.0,                    # 初始投資
        rolling_window=20,             # 20日滾動窗口
        base_penalty=0.5,              # 降低基礎懲罰係數
        market_diversity=True          # 強制市場多樣性
    )
    
    print(f"  狀態空間維度: {env.osn}")
    print(f"  動作空間: 連續 [0, 1]")
    print(f"  基礎懲罰係數: {env.base_penalty}")
    print(f"  市場多樣性: {'開啟' if env.market_diversity else '關閉'}")
    print(f"  市場模式分布: {dict(zip(env.market_modes, env.mode_probs))}")
    
    # 創建改進版DQL代理
    print("\n創建改進版DQL代理...")
    agent = EnhancedDQLAgent(
        env=env,
        hidden_units=128,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=32,
        target_update_freq=50
    )
    
    # 訓練代理
    print("\n開始訓練（確保市場多樣性）...")
    episodes = 100  # 增加訓練回合數
    rewards_history, market_modes = agent.learn(episodes, verbose=True)
    
    print(f"\n訓練完成！")
    print(f"  最終探索率: {agent.epsilon:.4f}")
    print(f"  平均獎勵 (最後10回合): {np.mean(rewards_history[-10:]):.2f}")
    print(f"  熊市訓練次數: {market_modes.count('bear')}")
    
    # 繪製改進版訓練結果
    print("\n繪製改進版訓練結果...")
    fig1 = plot_enhanced_training_results(rewards_history, market_modes)
    fig1.savefig('enhanced_training_progress.png', dpi=150, bbox_inches='tight')
    print("  已保存: enhanced_training_progress.png")
    
    # 測試代理（特別關注熊市表現）
    print("\n進行熊市壓力測試...")
    test_results = agent.test(episodes=5, test_mode='bear', verbose=True)
    
    print("\n進行混合市場測試...")
    mixed_results = agent.test(episodes=5, test_mode='mixed', verbose=True)
    
    # 繪製投資組合表現
    print("\n繪製投資組合表現...")
    fig2 = plot_adaptive_portfolio_performance(env)
    if fig2:
        fig2.savefig('adaptive_portfolio_performance.png', dpi=150, bbox_inches='tight')
        print("  已保存: adaptive_portfolio_performance.png")
    
    # 分析自適應策略
    print("\n進行策略分析...")
    stats = analyze_adaptive_strategy(env, agent)
    
    # 保存測試結果
    test_results.to_csv('bear_market_test_results.csv', index=False)
    mixed_results.to_csv('mixed_market_test_results.csv', index=False)
    print("\n測試結果已保存為CSV文件")
    
    plt.close('all')
    
    print("\n" + "=" * 80)
    print("改進總結報告")
    print("=" * 80)
    
    print("""
主要改進點：
===========

1. 市場多樣性增強：
   - 強制包含熊市場景（40%機率）
   - 增加均值回歸特性，避免單邊趨勢
   - 多樣化市場參數（漂移率、波動率）

2. 動態懲罰機制：
   - 基礎懲罰係數降低（從1.0降至0.5）
   - 高市場壓力時降低懲罰，鼓勵避險調整
   - 平穩市場時增加懲罰，避免過度交易

3. 增強獎勵函數：
   - 夏普比率考慮下行風險
   - 加入趨勢跟隨獎勵
   - 熊市避險獎勵
   - 動態風險調整

4. 改進狀態表示：
   - 技術指標（MA, RSI, MACD, 波動率）
   - 市場狀態分類（牛/熊/震盪）
   - 市場壓力指標
   - 歷史報酬率序列

5. 智能代理改進：
   - 更複雜的神經網絡結構
   - 自適應探索率（根據市場信心）
   - 目標網絡穩定訓練
   - 優先級經驗回放

預期效果：
=========

1. 避免"永遠做多"策略：
   - AI學會識別熊市並降低風險暴露
   - 在下跌趨勢中增加現金配置

2. 避免"躺平"策略：
   - 動態懲罰機制鼓勵必要時的調整
   - 市場壓力指標引導避險行為

3. 風險管理能力：
   - 根據波動率和市場壓力調整倉位
   - 平衡收益與風險的trade-off

4. 市場適應能力：
   - 在不同市場環境中表現穩定
   - 快速適應市場狀態變化

建議後續優化：
=============

1. 加入真實市場數據訓練
2. 實現多資產配置（股票、債券、黃金等）
3. 加入交易成本模型
4. 實現風險預算配置
5. 開發集成學習策略（多個agent投票）
    """)