from algorithm import BCND_algorithm


iter_num = 10
obs_dim =6
action_dim = 6
batch_size = 100
max_buffer_size = 1000
training_horizon = 100
learning_rate = 1e-4
num_networks = 3
network_config = dict(
    hidden_layer_dimension = 20,
    hidden_layer_numbers = 1
)
algo = BCND_algorithm(
    iter_num,
    obs_dim,
    action_dim,
    max_buffer_size,
    batch_size,
    training_horizon,
    learning_rate, 
    num_networks, 
    network_config
)
algo.train()