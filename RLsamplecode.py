import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from game.env import  *
import os
from scipy.spatial import distance
#from scipy.special import softmax
import pickle
os.chdir(r'C:\Users\SouwChuan\Documents\Work\QWOP\QWOP-RL-main')
LEARNING_RATE = 0.01
MEM_SIZE = 10000
DEVICE = torch.device("cpu")
SIMILARITYTHRESH = 0.00001
class cell(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.input_shape = 71
    self.action_space = 11

    self.fc1 = nn.Linear(self.input_shape, 64)
    self.fc2 = nn.Linear(64, 256)
    self.fc3 = nn.Linear(256, 128)
    self.fc4 = nn.Linear(128,32)
    self.fc5 = nn.Linear(32, self.action_space)

    self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
    self.loss = nn.MSELoss()
    self.to(DEVICE)
  
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x =self.fc5(x)


    return x
class ReplayBuffer:
    def __init__(self):
        self.mem_count = 0
        
        self.states = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=bool)
    
    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % MEM_SIZE
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1
    
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones
class Buffer:
    def __init__(self):
        self.mem_count = 0
        self.FailMEM = 50000
        self.discount = 0.9
        self.flat = 0.05
        #Need to store the state,action pair that are unsafe
        self.states = np.zeros((self.FailMEM, *env.observation_space.shape),dtype=np.float32)
        self.actions = np.zeros(self.FailMEM, dtype=np.int64)
        self.closeness = np.zeros(self.FailMEM, dtype=np.int64)
    def add(self, state, action):#pairs stored here must beforehand be a 
        #state that in worst case or following policy results in death
        mem_index = self.mem_count % self.FailMEM
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.closeness[mem_index] = -1
        self.mem_count += 1
    
    def checkdanger(self,currstate):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        #print(MEM_MAX)
        if MEM_MAX <= 20:
            return np.zeros(11)
        #print(self.states.shape)
        #print(currstate)
        #k+=2
        #try:
        currstate2 = np.reshape(currstate.numpy(),(1,71))
        try:
            distances = distance.cdist(currstate2, self.states[:MEM_MAX], "cosine")[0]
        except:
            print(currstate.numpy()[0])
            k+=1
        # except:
        #     print("asdasdasd",currstate)
        #for i in range(len(distances)):
            #distances[i] = (distances[i],i)
        #min_index = np.argmin(distances)
        danger = np.zeros(11)
        #print(MEM_MAX,distances)
        min_indexes = np.argpartition(distances, min(MEM_MAX,20))
        
        
        for id,i in enumerate(min_indexes):
            if distances[i]<SIMILARITYTHRESH:
                danger[self.actions[i]] = self.closeness[i]
        print(danger)
        return danger
    def addpossible(self,currstate,action,nextstate):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        if MEM_MAX <= 1:
            return 
        #print(nextstate)#nextstate[0].numpy()
        distances = distance.cdist(np.array(nextstate), self.states[:MEM_MAX], "cosine")[0]
        if np.min(distances)<SIMILARITYTHRESH:
            newclose = self.closeness[np.argmin(distances)]* self.discount-self.flat
            mem_index = self.mem_count % self.FailMEM
            self.states[mem_index]  = currstate
            self.actions[mem_index] = action
            self.closeness[mem_index] =newclose
            self.mem_count += 1
            return#state = env.reset()
    def addpossiblewin(self,currstate,action,nextstate):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        global startstate
        if MEM_MAX <= 1:
            return 
        #print(nextstate)#nextstate[0].numpy()
        if distance.cdist([nextstate],[startstate], "cosine")[0][0]<SIMILARITYTHRESH:
            mem_index = self.mem_count % self.FailMEM
            self.states[mem_index]  = state
            self.actions[mem_index] = action
            self.closeness[mem_index] =1
            self.mem_count += 1
            return
        distances = distance.cdist(np.array(nextstate), self.states[:MEM_MAX], "cosine")[0]
        if np.min(distances)<SIMILARITYTHRESH:
            newclose = self.closeness[np.argmin(distances)]* self.discount+self.flat
            mem_index = self.mem_count % self.FailMEM
            self.states[mem_index]  = state
            self.actions[mem_index] = action
            self.closeness[mem_index] =newclose
            self.mem_count += 1
            self.add(currstate,action)
            return
        
env = QWOPEnv()    
observation_space = env.observation_space.shape[0]
action_space = 11

EPISODES = 750

MEM_SIZE = 20000
BATCH_SIZE = 64
GAMMA = 0.99
EXPLORATION_MAX = 0.3
EXPLORATION_DECAY = 0.99
EXPLORATION_MIN = 0.005


class DQN_Solver:
    def __init__(self):
        self.totalq = 0
        self.qcount =0
        self.memory = ReplayBuffer()
        self.exploration_rate = EXPLORATION_MAX
        self.network = cell()
        self.failbuff = Buffer()
        self.winbuff = Buffer()
        

    def choose_action(self, observation):
    
        #if random.random() < self.exploration_rate:
            #return env.action_space.sample(),0
        
        state = torch.tensor(observation).float().detach()
        state = state.to(DEVICE)
        state = state.unsqueeze(0)
        q_values = self.network(state)
        #torch argmax returns the largest value respective index meaning this iswpnt probability based but rather value based
        #Therefore randomness through epsilon is needed for training.
        danger = self.failbuff.checkdanger(state)
        safes = self.winbuff.checkdanger(state)
        print(danger)
        soff = nn.Softmax()
        
        temp= torch.max(q_values)
        #temp = sum(q_values[0][0])
        #temp = max(temp,0.2)
        
        q_values= q_values.reshape(11)
        
        print(q_values.shape,q_values)
        for i in range(11):
            #print(danger[i],q_values)
            q_values[i] += danger[i]#+ safes[i]
        #print(q_values)
        #print(torch.argmax(q_values))
        q_values = soff(q_values)

        return torch.multinomial(q_values,1).item(),temp
    
    def learn(self):
        if self.memory.mem_count < BATCH_SIZE:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states , dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        states_ = torch.tensor(states_, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)#dones here means 0 if the crrent action ends simulation, and
        #1 if simulation continues
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        q_values = self.network(states)
        next_q_values = self.network(states_)
        
        predicted_value_of_now = q_values[batch_indices, actions]
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0]
        #print(predicted_value_of_future)
        #returns an array of batchsize of maximum predicted probability
        #K+=1
        #as usualy rewards is an array of 1s for this specific game.
        q_target = rewards + GAMMA * predicted_value_of_future * dones
        #important for dones to be inverted as otherwise NN will only learn from actiosn that immediately lose the game.
        #AS learnt gamma is used to learn importance of current actions on future rewards.
        #in this case predicted value of turue is the maximum 
        loss = self.network.loss(q_target, predicted_value_of_now)
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()
        for i in self.network.optimizer.param_groups:
            i["lr"] = i["lr"]*0.99
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def returning_epsilon(self):
        return self.exploration_rate


if False:
    with open("failsafesaves.pkl","rb") as inp:
        agent = pickle.load(inp)
        qarr = pickle.load(inp)
    start = agent.timer 
else:
    start = 1
startstate = env.reset()
def sigmoid(x):
    return 1/(1+np.exp(-x))

import matplotlib.pyplot as plt
for seednumber in range(10):
    torch.manual_seed(seednumber)
    agent = DQN_Solver()
    qarr = []
    rewardarr = []
    best_reward = 0
    average_reward = 0
    episode_number = []
    average_reward_number = []
    for i in range(start, EPISODES):
        #print(state)
        
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        score = 0
        episodebasedepsilon = EXPLORATION_MAX
        agent.exploration_rate = episodebasedepsilon
        totalq = 0
        qcount = 0
        totalreward= 0
        maxdistance = 0
        state = env.reset()
        while True:
            

            #agent.exploration_rate = max(EXPLORATION_MIN,sigmoid(time-maxdistance))
            #print(agent.exploration_rate)
            action,qval = agent.choose_action(state)
            totalq += qval
            qcount +=1
            #print(action)
            state_, reward, done, info = env.step(action)
            totalreward += reward
            
            state_ = np.reshape(state_, [1, observation_space])
            agent.failbuff.addpossible(state,reward,state_)
            agent.winbuff.addpossiblewin(state,reward,state_)
            agent.memory.add(state, action, reward, state_, done)
            agent.learn()
            state = state_
            score += reward

            if done:
                if type(totalq/qcount)== float:
                    qarr.append(totalq/qcount)
                else:
                    qarr.append((totalq/qcount).item())
                rewardarr.append(totalreward)
                

                
                agent.failbuff.add(state,action)
                state = env.reset()
                if score > best_reward:
                    best_reward = score
                    average_reward += score 
                    print("Episode {} Average Reward {} Best Reward {} Last Reward {} Epsilon {}".format(i, average_reward/i, best_reward, score, agent.returning_epsilon()))
                    break
                
            episode_number.append(i)
            average_reward_number.append(average_reward/i)
            if i%5 == 0:
                plt.rcParams["figure.autolayout"] = True
                #print(qarr)
                
            
                figure, axis = plt.subplots(2)
                axis[0].plot(range(len(qarr)),qarr)
                axis[0].set_title("qarr")
                axis[1].plot(range(len(rewardarr)),rewardarr)
                axis[1].set_title("rewardarr")
                plt.savefig(f"versions/stochastic/rewardvalues{seednumber}.png")
                plt.close()
                with open(f"versions/stochasticfailsafe/failsafesaves{seednumber}.pkl","wb") as outp:
                    agent.timer = i
                    pickle.dump(agent, outp, pickle.HIGHEST_PROTOCOL)
                    pickle.dump(qarr, outp, pickle.HIGHEST_PROTOCOL)
                    pickle.dump(rewardarr,outp,pickle.HIGHEST_PROTOCOL)
        if info:
            break
        episodebasedepsilon *= 0.99