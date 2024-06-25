import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple

#import plotly.graph_objects as go

import csv
import datetime as datetime
import time
import random




class Hyperparameters:
    EPISODE_NUM = 1000
    MAX_STEPS = 1000
    EPSILON = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995
    LEARNING_RATE = 0.001
    DISCOUNT_FACTOR = 0.99
    SOFT_UPDATE_TAU = 0.00099999
    MINIBATCH_SIZE = 64
    UPDATE_FREQUENCY = 4
    BUFFER_CAPACITY = 100000

LunarLanderenv = gym.make('LunarLander-v2')
LunarLanderenv.seed(42)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nodes = 64

class DDeepQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, seed):
        super(DDeepQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_dim, nodes)
        self.fc2 = nn.Linear(nodes, nodes)
        self.fc3 = nn.Linear(nodes, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ExperienceReplay:
    def __init__(self, action_dim, buffer_capacity, minibatch_size, seed):
        self.action_dim = action_dim
        self.memory = deque(maxlen=buffer_capacity)
        self.minibatch_size = minibatch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.minibatch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class DDQNAgent():
    def __init__(self, state_dim, action_dim, seed):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seed = random.seed(seed)


        self.local_qnetwork = DDeepQNetwork(state_dim, action_dim, seed).to(device)
        self.target_qnetwork = DDeepQNetwork(state_dim, action_dim, seed).to(device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=Hyperparameters.LEARNING_RATE)


        self.memory = ExperienceReplay(action_dim, Hyperparameters.BUFFER_CAPACITY, Hyperparameters.MINIBATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % Hyperparameters.UPDATE_FREQUENCY
        if self.t_step == 0:
            if len(self.memory) > Hyperparameters.MINIBATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, Hyperparameters.DISCOUNT_FACTOR)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_dim))

    def learn(self, experiences, discount_factor):
        states, actions, rewards, next_states, dones = experiences

        q_targets_next = self.local_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + discount_factor * q_targets_next * (1 - dones)
        q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.local_qnetwork, self.target_qnetwork, Hyperparameters.SOFT_UPDATE_TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def train_ddqn(episodes=Hyperparameters.EPISODE_NUM, epsilon=Hyperparameters.EPSILON, min_epsilon=Hyperparameters.EPSILON_MIN, epsilon_decay=Hyperparameters.EPSILON_DECAY, max_t_steps=Hyperparameters.MAX_STEPS):
    rewards = []
    rewards_average = deque(maxlen=100)
    eps = epsilon 
    lim = 0

    dqn_agent = DDQNAgent(state_dim=8, action_dim=4, seed=42)

    for episode in range(episodes):
        state = LunarLanderenv.reset()
        total_reward = 0

       

        for t in range(max_t_steps):

            delay = random.gauss(17, 8)
            while delay < 1:
                delay = random.gauss(17, 8)

            delay = float(delay)
            #time.sleep(delay/1000)


            #if(lim == 0):
            #    last_action = dqn_agent.act(state, eps)
            #    lim = 1
            
            if(delay>17):
                pass
            #    time.sleep(15/1000)
            #    action = last_action
                #action = LunarLanderenv.action_space.sample()
            #    next_state, reward, done, _ = LunarLanderenv.step(action)
            #    dqn_agent.step(state, action, reward, next_state, done)
            #    state = next_state
            #    total_reward += reward
            #    if done:
            #       break
            else:
            #   time.sleep(delay/1000)
            
                action = dqn_agent.act(state, eps)
                last_action = action
                next_state, reward, done, _ = LunarLanderenv.step(action)
                dqn_agent.step(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break

        rewards_average.append(total_reward) 
        rewards.append(total_reward) 
        eps = max(min_epsilon, epsilon_decay*eps) 

        print('\rEpisode--> {}\tTotal Reward: {:.2f}\tAverage Reward: {:.2f}\tEpsilon: {:.2f}'.format(episode, total_reward, np.mean(rewards_average), eps))

        time_now = datetime.datetime.now()

        csvfile.writerow([episode, total_reward, np.mean(rewards_average), time_now])
     

    return rewards

for i in range(2, 3):
    
    dados = ['episode', 'total_reward', 'media_reward', 'time']


    nome_file = "ddqn_d17_j08_c17_nothing_" + str(i)
    arq = open(nome_file, 'w')
    csvfile = csv.writer(arq)
    csvfile.writerow(dados)

    last_action = None
    # Run DDQN training
    training_scores = train_ddqn()


    #import matplotlib.pyplot as plt

    #episodes = list(range(Hyperparameters.EPISODE_NUM))
    #average_rewards = [np.mean(training_scores[max(0, i-100):(i+1)]) for i in range(Hyperparameters.EPISODE_NUM)]

    #plt.figure(figsize=(12, 6))
    #plt.plot(average_rewards, label='Average of last 100 episodes')
    #plt.plot(training_scores, alpha=0.5, label='Episode Reward')
    #plt.xlabel('Episodes')
    #plt.ylabel('Reward')
    #plt.title('Rewards and Moving Average Rewards per Episode')
    #plt.legend()
    #plt.grid(True)
    #plt.show()

    arq.close()