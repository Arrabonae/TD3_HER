import gymnasium as gym
import panda_gym
import numpy as np
from agent import DDPG
from buffer import HER
from config import *
from utils import plot_learning_curve

def train(memory, agent, env):
    n_steps = 0
    for i in range(EPOCHS):
        for _ in range(CYCLES):
            cycle_success, cycle_score = [], []
            for _ in range(EPISODES_PER_CYCLE):
                episode_score, episode_success = play_episode(memory, agent, env)
                cycle_success.append(episode_success)
                cycle_score.append(episode_score)
                for _ in range(OPTIMIZER_STEPS):
                    c_loss, a_loss = agent.learn(memory)
                    CRITIC_LOSS.append(c_loss)
                    ACTOR_LOSS.append(a_loss)
                    UPDATE_EPISODES.append(n_steps)
                    SUCCESS_HISTORY.append(episode_success)
                    SCORES_HISTORY.append(episode_score)
                    n_steps += 1
        agent.update_target_networks(TAU)
        print('Epoch: {}, Steps: {}  Score: {:.1f} Success: {:.3f}' .format(i, n_steps, episode_score, episode_success))



def play_episode(memory, agent, env):
    obs, info = env.reset()
    observation = obs['observation']
    achieved_goal = obs['achieved_goal']
    desired_goal = obs['desired_goal']
    done = False
    score = 0
    states, actions, rewards, states_, dones, d_goal, a_goal, a_goal_ = [], [], [], [], [], [], [], []

    while not done:

        action = agent.choose_action(np.concatenate([observation, desired_goal]))

        observation_, reward, done, truncated, info = env.step(action)

        if truncated:
            done = True

        states.append(observation)
        states_.append(observation_['observation'])
        rewards.append(reward)
        actions.append(action)
        dones.append(done)
        d_goal.append(observation_['desired_goal'])
        a_goal.append(achieved_goal)
        a_goal_.append(observation_['achieved_goal'])
        
        
        score += reward
        achieved_goal = observation_['achieved_goal']
        desired_goal = observation_['desired_goal']
        observation = observation_['observation']
       
    memory.store_episode(states, actions, rewards, states_, dones, d_goal, a_goal, a_goal_)
    success = info['is_success']

    return score, success



if __name__ == '__main__':
    env = gym.make("PandaPush-v3")
    obs_shape = env.observation_space['observation'].shape[0]
    goal_shape = env.observation_space['achieved_goal'].shape[0]
    n_actions=env.action_space.shape[0]
    
    memory = HER(obs_shape, n_actions,goal_shape, env.compute_reward)
    agent = DDPG(n_actions, env.action_space.low, env.action_space.high, [obs_shape+goal_shape])

    train(memory, agent, env)
    plot_learning_curve()


