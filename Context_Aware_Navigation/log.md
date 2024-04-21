costmap_v1: basic implementation 

    REPLAY_SIZE = 5000  # 10000 the size of replay buffer
    MINIMUM_BUFFER_SIZE = 1000  # 3000 the minimum size of replay buffer before training
    NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value
    COSTMAP_SCALE = 0.5 
    COSTMAP_RADIUS = 5
    COST_REWARD_COEFF = 1
    SAMPLE_DENSITY = 30 # 30 How many points will be generated for each axis

costmap_v2:  SAMPLE_DENSITY=100