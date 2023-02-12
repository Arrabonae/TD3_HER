import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
#locals
from networks import ActorNetwork, CriticNetwork
from config import *
from utils import OrnsteinUhlenbeckActionNoise


class DDPG:

    def __init__(self, n_actions, actions_low, actions_high, input_shape):

        self.n_actions = n_actions
        self.gamma = GAMMA
        self.input_shape = input_shape

        self.actor = ActorAgent(n_actions, actions_low, actions_high, alpha=ALPHA)
        self.critic = CriticAgent(beta=BETA)

    # def store_transition(self, obs, actions, obs_, reward, done):
    #     self.memory.store_transition(obs, actions, obs_, reward, done)

    def save_checkpoint(self):
        print('... saving checkpoint ...')

        self.actor.save_models()
        self.critic.save_models()

    def load_checkpoint(self):
        """
        Mo need to load critic agent, as it is only used for learning
        """
        print('... loading checkpoint ...')

        self.actor.load_models(self.input_shape)

    def update_network_parameters(self, tau):
        self.actor.update_network_parameters(tau)
        self.critic.update_network_parameters(tau)

    def choose_action(self, obs_goal, evaluate):
        """
        Environment takes parallel actions, so we need to return a dictionary of agents in the environment
        return such that {"agent_0": [action_0, action_1 ... action_n], "agent_1": [action_0, action_1 ... action_n], ...}
        """
        return self.actor.choose_action(obs_goal, evaluate)

    def learn(self, memory):

        if memory.mem_cntr < BATCH_SIZE:
            return 0, 0

        states, actions, rewards, states_, dones, desired_goals = memory.sample_memory()

        states = np.concatenate([states, desired_goals], axis=1)
        states_ = np.concatenate([states_, desired_goals], axis=1)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        states_ = tf.convert_to_tensor(states_, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)

        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        with tf.GradientTape() as tape:

            new_pi = self.actor.target_actor(states_)
            critic_value_ = tf.squeeze(self.critic.target_critic((states_, new_pi)), 1)
            critic_value = tf.squeeze(self.critic.critic((states, actions)), 1)

            target = rewards + self.gamma * critic_value_ * (1 - dones)
            target = tf.reduce_mean(target, axis=0)

            #OpenAI implementation uses huber loss, but after testing, MSE works better
            critic_loss = keras.losses.huber(target, critic_value, delta=1.0)
            #critic_loss = keras.losses.MSE(target, critic_value)


        critic_network_gradient = tape.gradient(critic_loss, self.critic.critic.trainable_variables)
        self.critic.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.critic.trainable_variables))
        
        with tf.GradientTape(persistent= True) as tape2:
            pi = self.actor.actor(states)
            actor_loss = -tf.squeeze(self.critic.critic((states, pi)),1)
            actor_loss = tf.reduce_mean(actor_loss, axis=0)

        actor_network_gradient = tape2.gradient(actor_loss, self.actor.actor.trainable_variables)
        self.actor.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.actor.trainable_variables))

        #self.update_network_parameters(TAU) 

        return critic_loss.numpy(), actor_loss.numpy()


class ActorAgent:
    """
    Class of the actor agent
    """
    def __init__(self, n_actions, actions_low, actions_high, alpha):
        
        self.n_actions = n_actions
        #for action selection: clipping / scaling the action to be between low and high
        self.actions_low = actions_low
        self.actions_high = actions_high

        #per OpenAI paper, Ornstein-Uhlenbeck process for action noise is the best to introduce exploration
        self.noise = OrnsteinUhlenbeckActionNoise(mu= np.zeros(self.n_actions))

        self.actor = ActorNetwork(n_actions=n_actions, name= 'Actor')
        self.target_actor = ActorNetwork(n_actions=n_actions, name= 'Target_actor')

        self.actor.compile(optimizer=Adam(learning_rate=alpha, clipnorm=1))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha, clipnorm=1))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        if targets == []:
            return
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

    def choose_action(self, obs, evaluate):
        state = tf.convert_to_tensor([obs], dtype=tf.float32)
        actions = self.actor(state)
        #add noise for exploration
        if not evaluate:
            actions += tf.convert_to_tensor(self.noise(), dtype=tf.float32)
        #clip the action to be between low and high otherwise environment will do it for you but 
        #it will affect performance and gives warning message
        actions = tf.clip_by_value(actions, self.actions_low, self.actions_high)
        return actions[0].numpy()

    def save_models(self):
        #print('... saving {} model ...' .format(self.actor.model_name))
        self.actor.save_weights(self.actor.checkpoint_file, save_format='h5')
        #print('... saving {} model ...' .format(self.target_actor.model_name))
        self.target_actor.save_weights(self.target_actor.checkpoint_file, save_format='h5')


    def load_models(self, actor_shape):
        print('... loading {} model ...'.format(self.actor.model_name))
        self.actor.build((BATCH_SIZE, actor_shape))
        self.actor.load_weights(self.actor.checkpoint_file)
        print('... loading {} model ...' .format(self.target_actor.model_name))
        self.target_actor.build((BATCH_SIZE, actor_shape))
        self.target_actor.load_weights(self.target_actor.checkpoint_file)


class CriticAgent():
    """
    Class of the critic agent
    """
    def __init__(self, beta):

        self.critic = CriticNetwork(name='Critic')
        self.target_critic = CriticNetwork(name='Target_critic')

        self.critic.compile(optimizer=Adam(learning_rate=beta, clipnorm=1))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta, clipnorm=1))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_critic.weights
        if targets == []:
            return
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def save_models(self):
        #print('... saving {} model ...'.format(self.critic.model_name))
        self.critic.save_weights(self.critic.checkpoint_file, save_format='h5')
        #print('... saving {} model ...'.format(self.target_critic.model_name))
        self.target_critic.save_weights(self.target_critic.checkpoint_file, save_format='h5')

    def load_models(self):
        print('... loading {} model ...'.format(self.critic.model_name))
        self.critic.load_weights(self.critic.checkpoint_file)
        print('... loading {} model ...'.format(self.target_critic.model_name))
        self.update_network_parameters(TAU)