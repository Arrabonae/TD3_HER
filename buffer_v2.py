import numpy as np
from config import *


class HER2:
    def __init__(self, input_shape, n_actions, goal_shape, compute_reward):
        self.memory_size = int(BUFFER_SIZE / T)
        self.mem_cntr = 0
        self.batch_size = BATCH_SIZE
        self.input_shape = input_shape
        self.compute_reward = compute_reward
        self.n_actions = n_actions
        self.goal_shape = goal_shape

        self.states = np.zeros((self.memory_size, self.input_shape),dtype=np.float32)
        self.states_ = np.zeros((self.memory_size, self.input_shape), dtype=np.float64)
        self.actions = np.zeros((self.memory_size, self.n_actions), dtype=np.float32)
        self.rewards = np.zeros(self.memory_size, dtype=np.float32)
        self.dones = np.zeros(self.memory_size, dtype=np.bool_)
        
        self.desired_goals = np.zeros((self.memory_size, self.goal_shape), dtype=np.float32)
        self.achieved_goals = np.zeros((self.memory_size, self.goal_shape), dtype=np.float32)
        self.achieved_goals_ = np.zeros((self.memory_size, self.goal_shape), dtype=np.float64)
        self.infos = np.zeros(self.memory_size, dtype=np.bool_)
        print("Running simple HER")


    def store_transition(self, state, action, reward, state_, done, d_goal, a_goal, a_goal_):
        
        i = self.mem_cntr % self.memory_size
        
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
            self.store_transition(state[i], action[i], reward[i], state_[i], done[i], d_goal[i], a_goal[i], a_goal_[i])
            for hindsight_goal in hindsight_goals[i]:
                hindsight_reward = self.compute_reward(a_goal_[i], hindsight_goal, {})
                self.store_transition(state[i], action[i], hindsight_reward, state_[i], done[i], hindsight_goal, a_goal[i], a_goal_[i])

    def sample_memory(self):    
        
        memory_max = min(self.mem_cntr, self.memory_size)
        batch = np.random.choice(memory_max, self.batch_size, replace=False)
        
        return  self.states[batch], \
                self.actions[batch], \
                self.rewards[batch],\
                self.states_[batch], \
                self.dones[batch],\
                self.desired_goals[batch]


    # def sample_memory(self):
    #     """
    #     Per OpenAI baselines
    #     I've adopted future strategy with k=4
    #     """
    #     future_p = 1 - (1. / (1 + REPLAY_K))

    #     # Select which episodes and time steps to use.  
    #     max = min(self.mem_cntr, self.memory_size)
    #     episode_samples = np.random.randint(0, max, self.batch_size)
    #     t_samples = np.random.randint(T, size=self.batch_size)

    #     sample_states = self.states[episode_samples, t_samples]
    #     sample_actions = self.actions[episode_samples, t_samples]
    #     sample_rewards = self.rewards[episode_samples, t_samples]
    #     sample_states_ = self.states_[episode_samples, t_samples]
    #     sample_dones = self.dones[episode_samples, t_samples]
    #     sample_desired_goals = self.desired_goals[episode_samples, t_samples]
    #     sample_achieved_goals = self.achieved_goals[episode_samples, t_samples]
    #     sample_infos = self.infos[episode_samples, t_samples]


    #     # Select future time indexes proportional with probability future_p. These
    #     # will be used for HER replay by substituting in future goals.
    #     her_indexes = np.where(np.random.uniform(size=self.batch_size) < future_p)
    #     future_offset = np.random.uniform(size=self.batch_size) * (T - t_samples)
    #     future_offset = future_offset.astype(int)
    #     future_t = (t_samples + future_offset)[her_indexes]

    #     # Replace goal with achieved goal but only for the previously-selected
    #     # HER transitions (as defined by her_indexes). For the other transitions,
    #     # keep the original goal.

    #     future_achieved_goal =  self.achieved_goals[episode_samples[her_indexes], future_t]
    #     sample_desired_goals[her_indexes] = future_achieved_goal
    #     future_infos = self.infos[episode_samples[her_indexes], future_t]
    #     sample_infos[her_indexes] = future_infos

    #     #  Re-compute reward since we may have substituted the goal.
    #     for idx, value in enumerate(sample_infos):
    #         sample_rewards[idx] = self.compute_reward(sample_achieved_goals[idx], sample_desired_goals[idx], {'is_success': value})

    #     assert(sample_states.shape == (self.batch_size, self.input_shape))
    #     assert(sample_actions.shape == (self.batch_size, self.n_actions))
    #     assert(sample_rewards.shape == (self.batch_size, ))
    #     assert(sample_states_.shape == (self.batch_size, self.input_shape))
    #     assert(sample_dones.shape == (self.batch_size, ))
    #     assert(sample_desired_goals.shape == (self.batch_size, self.goal_shape))

    #     return sample_states, sample_actions, sample_rewards, sample_states_, sample_dones, sample_desired_goals

    def ready(self):
        return self.mem_cntr > self.batch_size