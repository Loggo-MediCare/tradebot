import math
import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

def bsm_call_value(St, K, T, t, r, sigma):
    tau = T - t
    if tau <= 0:
        return max(St - K, 0)
    d1 = (math.log(St / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * math.sqrt(tau))
    d2 = d1 - sigma * math.sqrt(tau)
    call_value = St * stats.norm.cdf(d1) - K * math.exp(-r * tau) * stats.norm.cdf(d2)
    return call_value

def bsm_delta(St, K, T, t, r, sigma):
    tau = T - t
    if tau <= 0:
        return 1 if St > K else 0
    d1 = (math.log(St / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * math.sqrt(tau))
    return stats.norm.cdf(d1)

def simulate_gbm(S0, T, r, sigma, steps):
    dt = T / steps
    s = [S0]
    for _ in range(steps):
        st = s[-1] * math.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * random.gauss(0, 1))
        s.append(st)
    return np.array(s)

def option_replication(gbm, K, T, r, sigma):
    steps = len(gbm) - 1
    dt = T / steps
    bond = np.exp(r * np.arange(len(gbm)) * dt)
    res = pd.DataFrame()
    for i in range(steps):
        St = gbm[i]
        tau = T - i * dt
        C = bsm_call_value(St, K, T, i * dt, r, sigma)
        if i == 0:
            s = bsm_delta(St, K, T, i * dt, r, sigma)
            b = (C - s * St) / bond[i]
            V = C
        else:
            V = s * St + b * bond[i]
            s = bsm_delta(St, K, T, i * dt, r, sigma)
            b = (V - s * St) / bond[i]
        df = pd.DataFrame({'St': [St], 'C': [C], 'V': [V], 's': [s], 'b': [b]})
        res = pd.concat([res, df], ignore_index=True)
    return res

class observation_space:
    def __init__(self, n):
        self.shape = (n,)

class action_space:
    def __init__(self):
        pass
    def seed(self, seed):
        random.seed(seed)
    def sample(self):
        return random.random()

class Hedging:
    def __init__(self, S0, K_, T, r_, sigma_, steps):
        self.initial_value = S0
        self.strike_ = np.array(K_) if isinstance(K_, list) else np.array([K_])
        self.maturity = T
        self.short_rate_ = np.array(r_) if isinstance(r_, list) else np.array([r_])
        self.volatility_ = np.array(sigma_) if isinstance(sigma_, list) else np.array([sigma_])
        self.steps = steps
        self.observation_space = observation_space(8)
        self.action_space = action_space()
        self.portfolios = pd.DataFrame()
        self.episode = 0

    def _simulate_data(self):
        s = [self.initial_value]
        self.strike = random.choice(self.strike_)
        self.short_rate = random.choice(self.short_rate_)
        self.volatility = random.choice(self.volatility_)
        self.dt = self.maturity / self.steps
        for t in range(1, self.steps + 1):
            st = s[-1] * math.exp((self.short_rate - 0.5 * self.volatility ** 2) * self.dt + self.volatility * math.sqrt(self.dt) * random.gauss(0, 1))
            s.append(st)
        self.data = pd.DataFrame({'index': s})
        self.data['bond'] = np.exp(self.short_rate * np.arange(len(self.data)) * self.dt)

    def _get_state(self):
        St = self.data['index'].iloc[self.bar]
        Bt = self.data['bond'].iloc[self.bar]
        ttm = self.maturity - self.bar * self.dt
        if ttm > 0:
            Ct = bsm_call_value(St, self.strike, self.maturity, self.bar * self.dt, self.short_rate, self.volatility)
        else:
            Ct = max(St - self.strike, 0)
        return np.array([St, Bt, ttm, Ct, self.strike, self.short_rate, self.stock, self.bond]), {}

    def reset(self):
        self.bar = 0
        self.bond = 0
        self.stock = 0
        self.treward = 0
        self.episode += 1
        self._simulate_data()
        self.state, _ = self._get_state()
        return self.state, _

    def step(self, action):
        if self.bar == 0:
            reward = 0
            self.bar += 1
            self.stock = float(action)
            self.bond = (self.state[3] - self.stock * self.state[0]) / self.state[1]
            self.new_state, _ = self._get_state()
            done = False
        else:
            self.bar += 1
            self.new_state, _ = self._get_state()
            phi_value = self.stock * self.new_state[0] + self.bond * self.new_state[1]
            p1 = phi_value - self.new_state[3]
            df = pd.DataFrame({'e': [self.episode], 's': [self.stock], 'b': [self.bond], 'phi': [phi_value], 'C': [self.new_state[3]], 'p1': [p1], 'p1%': [p1 / max(self.new_state[3], 1e-4) * 100], 'St': [self.new_state[0]], 'Bt': [self.new_state[1]], 'K': [self.strike], 'r': [self.short_rate], 'sigma': [self.volatility]})
            self.portfolios = pd.concat([self.portfolios, df], ignore_index=True)
            reward = - (phi_value - self.new_state[3]) ** 2
            self.treward += reward
            self.stock = float(action)
            self.bond = (self.new_state[3] - self.stock * self.new_state[0]) / self.new_state[1]
            done = self.bar == len(self.data) - 1
        self.state = self.new_state
        return self.state, float(reward), done, False, {}

class Net(nn.Module):
    def __init__(self, input_size, hu):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hu)
        self.fc2 = nn.Linear(hu, hu)
        self.fc3 = nn.Linear(hu, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQLAgent:
    def __init__(self, state_size, hu=24, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=32):
        self.state_size = state_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.model = Net(state_size, hu)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

class HedgingAgent(DQLAgent):
    def opt_action(self, state):
        bounds = (0, 1)
        def f(x):
            x = x[0]
            s = state.copy()
            s[0, 6] = x
            s[0, 7] = (s[0, 3] - x * s[0, 0]) / s[0, 1]
            s_tensor = torch.tensor(s, dtype=torch.float32)
            return self.model(s_tensor).item()
        try:
            result = minimize(lambda x: -f(x), [0.5], bounds=[bounds], method='SLSQP')
            action = result['x'][0]
        except:
            action = 0.5
        return action

    def act(self, state):
        state = np.reshape(state, (1, -1))
        if random.random() <= self.epsilon:
            return self.env.action_space.sample()
        return self.opt_action(state)

    def remember(self, state, action, reward, next_state, done):
        state = np.reshape(state, (1, -1))
        next_state = np.reshape(next_state, (1, -1))
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                ns = next_state.copy()
                act = self.opt_action(ns)
                ns[0, 6] = act
                ns[0, 7] = (ns[0, 3] - act * ns[0, 0]) / ns[0, 1]
                ns_tensor = torch.tensor(ns, dtype=torch.float32)
                target += self.gamma * self.model(ns_tensor).item()
            state_tensor = torch.tensor(state, dtype=torch.float32)
            pred = self.model(state_tensor)
            target_tensor = torch.tensor([[target]], dtype=torch.float32)
            loss = self.loss_fn(pred, target_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes):
        for e in range(episodes):
            state, _ = self.env.reset()
            state = np.reshape(state, (1, -1))
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = np.reshape(next_state, (1, -1))
                self.remember(state, action, reward, next_state, done)
                state = next_state
            self.replay()
            print(f'Episode {e}: Total Penalty {self.env.treward}')

    def test(self, episodes):
        for e in range(episodes):
            state, _ = self.env.reset()
            state = np.reshape(state, (1, -1))
            done = False
            total_penalty = 0
            while not done:
                action = self.opt_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = np.reshape(next_state, (1, -1))
                total_penalty -= reward
                state = next_state
            print(f'Total Penalty: {total_penalty:.2f}')

# Example usage
plt.style.use('seaborn-v0_8-darkgrid')

random.seed(100)
torch.manual_seed(100)

S0 = 100
hedging = Hedging(S0, K_=[90,95,100,105,110], T=1.0, r_=[0,0.01,0.05], sigma_=[0.1,0.15,0.2], steps=504)
agent = HedgingAgent(state_size=8, hu=24, lr=0.001)
agent.env = hedging
# agent.train(100)  # Uncomment to train
# agent.test(10)  # Uncomment to test
print("Code is runnable!")
