import numpy as np
import matplotlib.pyplot as plt
from config import *

def plot_learning_curve():
    x = [i+1 for i in range(EPOCHS)]        
    _, ax1 = plt.subplots()
    ax1.plot(x, SCORES_HISTORY, label = 'Episode reward', color= 'green')
    ax1.plot(x, SUCCESS_HISTORY, label = 'Success rate', color= 'blue')
    ax1.set_ylabel("Score / Success rate")
    ax1.set_xlabel("Episodes")
    ax1.legend()
    plt.title('Performance of the agents')
    plt.savefig(FIGURE_FILE)
    plt.clf()

    _, ax2 = plt.subplots()
    ax2.plot(UPDATE_EPISODES, CRITIC_LOSS, label='Critic loss', color='blue')
    ax2.plot(UPDATE_EPISODES, ACTOR_LOSS, label='Actor loss', color='green')
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Episodes")
    ax2.legend()
    plt.savefig(FIGURE_FILE2)
    plt.clf()

    x = [i+1 for i in range(len(EPISODE_LENGTH))]   
    _, ax3 = plt.subplots()
    ax3.plot(x, EPISODE_LENGTH, label='Episode Length', color='blue')
    ax3.set_ylabel("Episode Length")
    ax3.set_xlabel("Episodes")
    ax3.legend()
    plt.savefig(FIGURE_FILE3)


class OrnsteinUhlenbeckActionNoise():
    """
    OpenAI baselines implementation of Ornstein-Uhlenbeck process
    """
    def __init__(self, mu):
        self.theta = THETA
        self.mu = mu
        self.sigma = SIGMA
        self.dt = DT
        self.x0 = X0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)