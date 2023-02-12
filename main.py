import gymnasium as gym
import panda_gym
import numpy as np
from agent import DDPG
from buffer import HER
from buffer_v2 import HER2
from config import *
from utils import plot_learning_curve

def train(memory, agent, env):
    n_steps, best_success = 0, 0
    for i in range(EPOCHS):
        epoch_success, epoch_score = [], []
        for c in range(CYCLES):
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
                n_steps += 1
            agent.update_network_parameters(TAU)
            epoch_success.append(np.mean(cycle_success))
            epoch_score.append(np.mean(cycle_score))
        test_score, test_success = [], []
        for episode in range(N_TESTS):
            episode_score, episode_success = play_episode(memory, agent, env, evaluate=True)
            test_success.append(episode_success)
            test_score.append(episode_score)

        SCORES_HISTORY.append(np.mean(test_score))
        SUCCESS_HISTORY.append(np.mean(test_success))
        print('Epoch: {} Score: {:.1f}; Success: {:.2%}' .format(i, np.mean(test_score), np.mean(test_success)))

def play_episode(memory, agent, env, evaluate=False):
    obs, info = env.reset()
    observation = obs['observation']
    achieved_goal = obs['achieved_goal']
    desired_goal = obs['desired_goal']
    done = False
    score = 0
    states, actions, rewards, states_, dones, d_goal, a_goal, a_goal_, infos = [], [], [], [], [], [], [], [], []
    
    while not done:

        action = agent.choose_action(np.concatenate([observation, desired_goal]), evaluate)

        observation_, reward, done, truncated, info = env.step(action)

        # if truncated:
        #     done = True

        states.append(observation)
        states_.append(observation_['observation'])
        rewards.append(reward)
        actions.append(action)
        dones.append(done)
        d_goal.append(observation_['desired_goal'])
        a_goal.append(achieved_goal)
        a_goal_.append(observation_['achieved_goal'])
        infos.append(info['is_success'])
        
        score += reward
        achieved_goal = observation_['achieved_goal']
        desired_goal = observation_['desired_goal']
        observation = observation_['observation']
    
    
    EPISODE_LENGTH.append(len(states))
    if not evaluate:
        #Openai HER
        memory.store_transition(states, actions, rewards, states_, dones, d_goal, a_goal, a_goal_, infos)
        
        #Simple HER
        #memory.store_episode(states, actions, rewards, states_, dones, d_goal, a_goal, a_goal_)
    
    success = info['is_success']

    return score, success



if __name__ == '__main__':
    env = gym.make("PandaReach-v3")
    obs_shape = env.observation_space['observation'].shape[0]
    goal_shape = env.observation_space['achieved_goal'].shape[0]
    n_actions=env.action_space.shape[0]
    
    memory = HER(obs_shape, n_actions, goal_shape, env.compute_reward)
    #memory = HER2(obs_shape, n_actions, goal_shape, env.compute_reward)
    agent = DDPG(n_actions, env.action_space.low, env.action_space.high, [obs_shape+goal_shape])

    train(memory, agent, env)
    plot_learning_curve()


