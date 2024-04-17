costmap_v1: basic implementation 
    REPLAY_SIZE = 5000  # 10000 the size of replay buffer
    MINIMUM_BUFFER_SIZE = 1000  # 3000 the minimum size of replay buffer before training
    BATCH_SIZE = 16  # 64 the size of mini-batch used to train the network. sample from replay buffer
    INPUT_DIM = 8  # 7 the dimension of policy network input
    EMBEDDING_DIM = 128  # the dimension of embedding vector
    NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value
    NORMALIZE_UTILITY = False  # do you want to normalize the utility of nodes
    K_SIZE = 20  # the number of neighboring nodes
    USE_GPU = True  # do you want to collect training data using GPUs
    USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
    NUM_GPU = 1  # the number of GPUs you want to use
    NUM_META_AGENT = 4  # 24 the number of meta agents
    LR = 1e-5  # learning rate
    GAMMA = 1  # discount factor
    DECAY_STEP = 256  # not use
    SUMMARY_WINDOW = 50
    LOAD_MODEL = False  # do you want to load the model trained before
    SAVE_IMG_GAP = 500  # 500 save image every SAVE_IMG_GAP episodes
    COSTMAP_SCALE = 0.5 
    COSTMAP_RADIUS = 5
    COST_REWARD_COEFF = 1
    SAMPLE_DENSITY = 30 # 30 How many points will be generated for each axis

costmap_v2:  SAMPLE_DENSITY=100