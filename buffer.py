import numpy as np
from config import *


class HER:
    def __init__(self, input_shape, n_actions, goal_shape, compute_reward):
        self.buffer_size = BUFFER_SIZE
        self.mem_cntr = 0
        self.batch_size = BATCH_SIZE
        self.input_shape = input_shape
        self.compute_reward = compute_reward
        self.n_actions = n_actions
        self.goal_shape = goal_shape

        self.states = np.zeros((self.buffer_size, self.input_shape),dtype=np.float32)
        self.states_ = np.zeros((self.buffer_size, self.input_shape), dtype=np.float64)
        self.actions = np.zeros((self.buffer_size, self.n_actions), dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.bool_)
        
        self.desired_goals = np.zeros((self.buffer_size, self.goal_shape), dtype=np.float32)
        self.achieved_goals = np.zeros((self.buffer_size, self.goal_shape), dtype=np.float32)
        self.achieved_goals_ = np.zeros((self.buffer_size, self.goal_shape), dtype=np.float64)


    def store_memory(self, state, action, reward, state_, done, d_goal, a_goal, a_goal_):
        
        i = self.mem_cntr % self.buffer_size
        
        self.states[i] = state
        self.states_[i] = state_
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.desired_goals[i] = d_goal
        self.achieved_goals[i] = a_goal
        self.achieved_goals_[i] = a_goal_
        
        self.mem_cntr += 1
    
    def store_episode(self, state, action, reward, state_, done, d_goal, a_goal, a_goal_):

        hindsight_goals = [[a_goal_[-1]]] * len(a_goal_)
        
        for i in range(len(state)):
            self.store_memory(state[i], action[i], reward[i], state_[i], done[i], d_goal[i], a_goal[i], a_goal_[i])
            for goal in hindsight_goals[i]:

                hindsight_reward = self.compute_reward(a_goal_[i], goal, dict())
                self.store_memory(state[i], action[i], hindsight_reward, state_[i], done[i], goal, a_goal[i], a_goal_[i])

    def sample_memory(self):    
        
        memory_max = min(self.mem_cntr, self.buffer_size)
        batch = np.random.choice(memory_max, self.batch_size, replace=False)
        
        return  self.states[batch], \
                self.actions[batch], \
                self.rewards[batch],\
                self.states_[batch], \
                self.dones[batch],\
                self.desired_goals[batch]
    
    def ready(self):
        return self.mem_cntr > self.batch_size