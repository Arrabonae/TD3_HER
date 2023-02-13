import gymnasium as gym
import panda_gym
import numpy as np
from agent import TD3
from buffer import HER
from config import *
from utils import plot_learning_curve

def train(memory, agent, env):
    best_success, best_score = -np.inf, -np.inf
    for i in range(EPOCHS):
        for c in range(CYCLES):
            for ec in range(EPISODES_PER_CYCLE):
                _, _ = play_episode(memory, agent, env)
                #print("Playing: ", ec)
            for o in range(OPTIMIZER_STEPS):
                agent.learn(memory)
                #print("Learning: ", o)
            agent.update_network_parameters(TAU)
            #print("Cycle: ", c)
        test_score, test_success = [], []
        for episode in range(N_TESTS):
            score, success = play_episode(memory, agent, env, evaluate=True)
            test_success.append(success)
            test_score.append(score)

        if np.mean(test_score) > best_score:
            best_success = np.mean(test_success)
            best_score = np.mean(test_score)
            agent.save_checkpoint()
            print('Best success so far: {:.2%}; best score so far: {:.1f}' .format(best_success, best_score))

        SCORES_HISTORY.append(np.mean(test_score))
        SUCCESS_HISTORY.append(np.mean(test_success))
        print('Epoch: {} Score: {:.1f}; Success: {:.2%}' .format(i, np.mean(test_score), np.mean(test_success)))

def play_episode(memory, agent, env, evaluate=False):
    obs, info = env.reset()
    #obs = env.reset()
    observation = obs['observation']
    achieved_goal = obs['achieved_goal']
    desired_goal = obs['desired_goal']
    done = False
    score = 0
    states, actions, rewards, states_, dones, d_goal, a_goal, a_goal_, infos = [], [], [], [], [], [], [], [], []
    
    while not done:

        action = agent.choose_action(np.concatenate([observation, desired_goal]), evaluate)

        #observation_, reward, done, info = env.step(action)
        observation_, reward, done, truncated, info = env.step(action)

        states.append(observation)
        states_.append(observation_['observation'])
        rewards.append(reward)
        actions.append(action)
        dones.append(done)
        d_goal.append(desired_goal)
        a_goal.append(achieved_goal)
        a_goal_.append(observation_['achieved_goal'])
        infos.append(info['is_success'])
        
        score += reward
        achieved_goal = observation_['achieved_goal']
        observation = observation_['observation']
    
        if truncated:
            done = True

    if not evaluate:
        #Openai HER
        memory.store_transition(states, actions, rewards, states_, dones, d_goal, a_goal, a_goal_, infos)
    
    success = info['is_success']

    return score, success


if __name__ == '__main__':
    env = gym.make("PandaReach-v3")
    obs_shape = env.observation_space['observation'].shape[0]
    goal_shape = env.observation_space['achieved_goal'].shape[0]
    n_actions=env.action_space.shape[0]
    
    memory = HER(obs_shape, n_actions, goal_shape, env.compute_reward)
    agent = TD3(n_actions, env.action_space.low, env.action_space.high, [obs_shape+goal_shape])

    train(memory, agent, env)
    plot_learning_curve()


