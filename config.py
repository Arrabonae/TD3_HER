#General
EPOCHS = 50
CYCLES = 50
EPISODES_PER_CYCLE = 16
OPTIMIZER_STEPS = 40
N_TESTS = 10
FIGURE_FILE = 'plots/learning_curve_complex_v4.png'
FIGURE_FILE2 = 'plots/loss_complex_v4.png'
FIGURE_FILE3 = 'plots/episode_length_complex_v4.png'
CHECKPOINT_DIR = 'models/'
SCORES_HISTORY = []
SUCCESS_HISTORY = []    
UPDATE_EPISODES = []
EPISODE_LENGTH = []
CRITIC_LOSS = []
ACTOR_LOSS = []

#Memory
BATCH_SIZE = 128
BUFFER_SIZE = 10**6
T = 50 #size of one episode
REPLAY_K = 4

#Training
ALPHA = 0.0001
BETA = 0.001
TAU = 0.95
GAMMA = 0.98

#Network architecture
WEIGHT_INIT = 'he_normal'
BIAS_INIT = 'he_normal'
CRITIC_DENSE1 = 64
CRITIC_DENSE2 = 64
CRITIC_DENSE3 = 64
ACTORS_DENSE1 = 64
ACTORS_DENSE2 = 64
ACTORS_DENSE3 = 64

CRITIC_ACTIVATION_HIDDEN = 'relu'
CRITIC_ACTIVATION_OUTPUT = None
ACTORS_ACTIVATION_HIDDEN = 'relu'
ACTORS_ACTIVATION_OUTPUT = 'tanh' #Action low and high are -2 and 2, we need to scale the output of the network

#Ornstein-Uhlenbeck process
THETA = 0.15
SIGMA = 0.2
DT = 1e-2
X0 = None