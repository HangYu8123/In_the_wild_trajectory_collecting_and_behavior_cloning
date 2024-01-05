import numpy as np
from replay_buffer import SimpleReplayBuffer
from network import BCND_network
import torch
import utils.bc_utils
import torch.nn.functional as functional

DEVICE = utils.bc_utils.device()

class BCND_Trainer():
    def __init__(
            self,
            obs_dim,
            action_dim,
            batch_size,
            training_horizon,
            learning_rate:float,
            num_networks:int,
            network_config:dict,
            
            ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = network_config["hidden_layer_dimension"]
        self.hidden_layers = network_config["hidden_layer_numbers"] 
        self.batch_size = batch_size  
        self.num_policies = num_networks 
        self.lr = learning_rate   
        # flatten to a number if in_dim or action_dim is high-dimensional 
        if not isinstance(obs_dim, int):
            self.obs_dim = np.prod(obs_dim)
        if not isinstance(action_dim, int):
            self.action_dim = np.prod(action_dim)

        self.replay_buffer = SimpleReplayBuffer()
        self.optimizer = torch.optim.Adam
        # self.policy = BCND_network(self.obs_dim, self.action_dim, self.hidden_dim, self.hidden_layers)
        # self.policy.to(DEVICE)

        # a list of policies
        self.policies = []
        # old policies for rewards
        self.old_policies = []
        # initialize old policies as random networks
        for _ in range(num_networks):
            network_k = BCND_network(self.obs_dim, self.action_dim, self.hidden_dim, self.hidden_layers)
            self.old_policies.append(network_k)

            # TODO:add function to load dataset


    # BCND reward from old policies and sampled experiences
    def reward(self, observations, actions:np.ndarray):
        rewards = []
        actions_tensor = torch.tensor(actions).to(DEVICE)
        for i in range(self.num_policies):
            reward_k_mean, reward_k_std:torch.Tensor = self.old_policies[i](observations)
            reward = torch.distributions.Normal(reward_k_mean, reward_k_std)
            assert actions_tensor.size == reward_k_mean.size
            log_prob_old:torch.Tensor = -reward.log_prob(actions_tensor)
            prob_old = log_prob_old.exp()
            rewards.append(prob_old)
        reward_tensor =torch.stack(rewards)
        reward_mean = reward_tensor.mean(dim=0,keepdim=False)
        return reward_mean.detach()
    

        
            



    def run_batch(self, policy:BCND_network):
        batch = self.replay_buffer.random_sample(self.batch_size)
        observations:np.ndarray = batch["observations"]
        actions = batch["actions"]
        assert np.size(observations, axis=1) == self.obs_dim, "ERROR: Observations from replay buffer have wrong dimension!\n"
        assert np.size(actions, axis = 1) == self.action_dim, "ERROR: Actions from replay buffer have wrong dimension!\n"
        observations = torch.tensor(observations).to(DEVICE)
        actions = torch.tensor(actions).to(DEVICE)
        predicted_action_mean, predicted_action_std:torch.Tensor = policy(observations)
        policy_dist = torch.distributions.Normal(predicted_action_mean,predicted_action_std)
        assert actions.size == predicted_action_mean.size
        log_prob_action_n = policy_dist.log_prob(actions)
        rewards = self.reward(observations, actions)
        loss = -log_prob_action_n * rewards
        optimizer = self.optimizer(policy.parameters(), self.lr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



