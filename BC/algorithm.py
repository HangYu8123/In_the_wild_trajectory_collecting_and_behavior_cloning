from agent import BCND_Trainer
from utils.bc_utils import get_all_bag_files,whole_bag_to_messages, read_one_trajectory_to_each_buffer





class BCND_algorithm():
    def __init__(self, 
                 iterartion_num,
                obs_dim,
            action_dim,            
            max_buffer_size,
            batch_size,
            training_horizon,
            learning_rate:float,
            num_networks:int,
            network_config:dict):
        
        self.trainer = BCND_Trainer(obs_dim,
            action_dim,            
            max_buffer_size,
            batch_size,
            training_horizon,
            learning_rate,
            num_networks,
            network_config)
        
        self.iteration_num = iterartion_num
        # load trajectory data to buffer
        ls = get_all_bag_files()
        msgs = whole_bag_to_messages(ls)
        read_one_trajectory_to_each_buffer(num_networks,self.trainer.buffers, msgs)

    def train(self):
        for _ in range(self.iteration_num):
            self.trainer.run_one_iterarion()


        