FOLDER_NAME = "costmap_v4"
model_path = f"model/{FOLDER_NAME}"
train_path = f"train/{FOLDER_NAME}"
gifs_path = f"gifs/{FOLDER_NAME}"

MAX_EPISODE = 20000
REPLAY_SIZE = 4000  # 10000 the size of replay buffer
MINIMUM_BUFFER_SIZE = 1000  # 3000 the minimum size of replay buffer before training
NODE_PADDING_SIZE = 360  # 360 the number of nodes will be padded to this value
BATCH_SIZE = 16  # 64 the size of mini-batch used to train the network. sample from replay buffer
SAMPLE_DENSITY = 30  # 30 How many points will be generated for each axis
USE_GPU = True  # do you want to collect training data using GPUs
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 1  # the number of GPUs you want to use

INPUT_DIM = 8  # 7 the dimension of policy network input
EMBEDDING_DIM = 128  # the dimension of embedding vector
NORMALIZE_UTILITY = False  # do you want to normalize the utility of nodes
K_SIZE = 20  # the number of neighboring nodes
NUM_META_AGENT = 4  # 24 the number of meta agents

COSTMAP_SCALE = 0.2 # 0.2
COSTMAP_RADIUS = 10 # 5
COST_REWARD_COEFF = 5
LR = 1e-5  # learning rate
GAMMA = 1  # discount factor
DECAY_STEP = 256  # not use
MAX_STEP=128

SUMMARY_WINDOW = 50
LOAD_MODEL = False  # do you want to load the model trained before
SAVE_IMG_GAP = 100  # 500 save image every SAVE_IMG_GAP episodes
