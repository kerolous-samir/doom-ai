# Import Libraries

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import gym
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete # it's not working anymore because a companies distribution or smth like that

from src import experience_replay, image_preprocessing # already were ready in the course I didn't make them 


__all__ = ['SkipWrapper'] # there was an import for it with gym.wrappers but in the new verision not there anymore so I got it independently 

def SkipWrapper(repeat_count):
    class SkipWrapper(gym.Wrapper):
        """
            Generic common frame skipping wrapper
            Will perform action for `x` additional steps
        """
        def __init__(self, env):
            super(SkipWrapper, self).__init__(env)
            self.repeat_count = repeat_count
            self.stepcount = 0

        def _step(self, action):
            done = False
            total_reward = 0
            current_step = 0
            while current_step < (self.repeat_count + 1) and not done:
                self.stepcount += 1
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                current_step += 1
            if 'skip.stepcount' in info:
                raise gym.error.Error('Key "skip.stepcount" already in info. Make sure you are not stacking ' \
                                      'the SkipWrapper wrappers.')
            info['skip.stepcount'] = self.stepcount
            return obs, total_reward, done, info

        def _reset(self):
            self.stepcount = 0
            return self.env.reset()

    return SkipWrapper

# Build Brain

class CNN(nn.Module):
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size= 5 )
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size= 3 )
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size= 2 )
        self.fc1 = nn.Linear(in_features = self.count_neurons((1,80,80)),out_features= 40)
        self.fc2 = nn.Linear(in_features = 40,out_features= number_actions)
    
    def count_neurons(self, img_dim):
        x = Variable(torch.rand(1, *img_dim))
        x = F.relu(F.max_pool2d(self.conv1(x),3,2))
        x = F.relu(F.max_pool2d(self.conv2(x),3,2))
        x = F.relu(F.max_pool2d(self.conv3(x),3,2))
        return x.data.view(1,-1).size(1)
    
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x),3,2))
        x = F.relu(F.max_pool2d(self.conv2(x),3,2))
        x = F.relu(F.max_pool2d(self.conv3(x),3,2)) 
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    
# Build Body

class SoftmaxBody(nn.Module):
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T
        
    def forward(self, outputs):
        probs = F.softmax(outputs * self.T)
        action = probs.multinomal()
        return action


# Build AI    

class AI:
    def __init__(self,brain,body):
        self.brain = brain
        self.body = body
        
    def __call__(self,inputs):
        input = Variable(torch.from_numpy(np.array(inputs, dtype= np.float32())))
        output = self.brain(input)
        action = self.body(output)
        return action.data.numpy()


# Implementing Environment

doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)
number_actions = doom_env.action_space.n



cnn = CNN(number_actions)
body = SoftmaxBody(T= 1.0)
ai = AI(brain = cnn, body = body)

n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_steps = 10)
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000)


# Implementing Eligiblity Trace

def eligiblity_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = Variable(torch.from_numpy(np.array(series[0].state,series[-1].state, dtype= np.float32())))
        output = cnn(input)
        cumural_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumural_reward = step.reward + gamma * cumural_reward
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumural_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs,dtype= np.float32())) , torch.stack(targets)


# Implementing Moving Average

class MA:
    
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size
        
    def add(self,rewards):
        if isinstance(rewards,list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
    
    def average(self):
        return np.mean(self.list_of_rewards)
    
ma = MA(100)

loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr= 0.001)
nb_epochs = 100


# Training the AI

for epoch in range (1, nb_epochs +1):
    memory.run_steps(200)
    for batch in memory.sample_batch(128):
        inputs , targets = eligiblity_trace(batch)
        inputs , targets = Variable(inputs) , Variable(targets)
        predictions = cnn(inputs)
        loss_error = loss(predictions, targets)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    average_reward = ma.average()
    print("Epochs: %s, average reward: %s" %(str(epoch), str(average_reward)))

