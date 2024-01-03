import torch
import torch.nn as nn


class BCND_network(nn.Module):
    def __init__(self,
                 in_dim, 
                 out_dim,
                 hidden_dim=100, 
                 num_hidden_layers=2) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.out_dim = out_dim
        # building network
        self.layers = []
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(self.num_hidden_layers):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        self.network = nn.Sequential(*self.layers)
    
    def forward(self, data:torch.Tensor):
        # make a batch if data is a single vector
        if len(data.size()) == 1:
            data.unsqueeze(0)
    
        output = self.network(data)
        return output
    
    def train(self):
        super().train()
    def eval(self):
        super().eval()    

