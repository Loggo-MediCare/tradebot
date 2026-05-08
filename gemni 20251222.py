import numpy as np
import tensorflow as tf
from collections import deque
import random

# =========================================================
# 3. 高勝率優化環境 (High Win-Rate Environment)
# =========================================================
class HighPrecisionEnvironment:
    def __init__(self, df, feature_cols, initial_balance=10000, window_size=10):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_balance = initial_balance
        self.window_size = window_size 
        self.transaction_cost = 0.001425 # 台灣手續費
        self.sell_tax = 0.003            # 證交稅
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.total_shares = 0
        self.current_step = self.window_size
        self.trades_history = []
        self.portfolio_value_history = [self.initial_balance]
        self.max_net_worth = self.initial_balance
        return self._get_state()

    def _get_state(self):
        # 提取市場特徵 [cite: 33, 40]
        start = self.current_step - self.window_size
        market_state = self.df[self.feature_cols].iloc[start:self.current_step].values.flatten()

        # 核心：加入風險感知特徵 (回撤與持倉比)
        current_price = float(self.df['Close'].iloc[self.current_step])
        net_worth = self.balance + (self.total_shares * current_price)
        drawdown = (self.max_net_worth - net_worth) / self.max_net_worth if self.max_net_worth > 0 else 0
        
        account_state = np.array([
            self.balance / self.initial_balance,
            (self.total_shares * current_price) / self.initial_balance,
            net_worth / self.initial_balance,
            drawdown  # 讓模型學會「怕痛」，進而提升勝率
        ])
        return np.concatenate((market_state, account_state))

    def step(self, action):
        current_price = float(self.df['Close'].iloc[self.current_step])
        prev_net_worth = self.balance + (self.total_shares * current_price)
        
        # 執行動作
        if action == 1: # BUY
            shares_to_buy = int((self.balance * 0.95) / (current_price * (1 + self.transaction_cost)))
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.balance -= (cost * (1 + self.transaction_cost))
                self.total_shares += shares_to_buy
                self.trades_history.append(('BUY', self.current_step, current_price))

        elif action == 2: # SELL
            if self.total_shares > 0:
                revenue = self.total_shares * current_price
                self.balance += (revenue * (1 - self.transaction_cost - self.sell_tax))
                self.total_shares = 0
                self.trades_history.append(('SELL', self.current_step, current_price))

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # 計算獎勵：結合收益率與風險懲罰 
        new_price = float(self.df['Close'].iloc[self.current_step])
        net_worth = self.balance + (self.total_shares * new_price)
        self.max_net_worth = max(self.max_net_worth, net_worth)

        # 高勝率獎勵邏輯：放大虧損懲罰，鼓勵穩定獲利
        reward = (net_worth - prev_net_worth) / prev_net_worth * 100
        if reward < 0:
            reward *= 2.0  # 嚴厲懲罰虧損，迫使模型追求更高勝率
        
        return self._get_state(), reward, done, net_worth

# =========================================================
# 4. Double DQN 代理人 (Precision Double DQN Agent)
# =========================================================
class PrecisionDQNAgent:
    def __init__(self, state_size, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001 # 較低學習率以確保穩定性 [cite: 32, 44]
        
        # 雙網絡：減少預測偏差
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # 使用 Dueling 架構思想的深度網絡
        inputs = tf.keras.layers.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x) # 穩定輸入分布
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x) # 防止過擬合
        
        # 輸出層
        outputs = tf.keras.layers.Dense(self.action_size, activation='linear')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='huber', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size: return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([m[0] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        
        # Double DQN 核心：主網絡選動作，目標網絡算價值
        target_q = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0) # Online model
        next_target_q = self.target_model.predict(next_states, verbose=0) # Target model
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                target_q[i][action] = reward
            else:
                best_action = np.argmax(next_q_values[i])
                target_q[i][action] = reward + self.gamma * next_target_q[i][best_action]
        
        self.model.fit(states, target_q, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay