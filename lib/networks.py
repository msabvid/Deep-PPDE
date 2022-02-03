import torch
import torch.nn as nn
from collections import namedtuple
from typing import Tuple

class FFN(nn.Module):

    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()

        layers = []
        for j in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[j], sizes[j+1]))
            if j<(len(sizes)-2):
                layers.append(activation())
            else:
                layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad=False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad=True

    def hard_update(self, source_net):
        """Updates the network parameters by copying the parameters
        of another network
        """
        for target_param, source_param in zip(self.parameters(), source_net.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, source_net, tau):
        """Updates the network parameters with a soft update by polyak averaging
        """
        for target_param, source_param in zip(self.parameters(), source_net.parameters()):
            target_param.data.copy_((1-tau)*target_param.data + tau*source_param.data)
    
    def forward(self, *args):
        x = torch.cat(args,-1)
        return self.net(x)



class RNN(nn.Module):

    def __init__(self, rnn_in, rnn_hidden, ffn_sizes, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()
        self.rnn = nn.LSTM(input_size=rnn_in, hidden_size=rnn_hidden,
                num_layers=1,
                batch_first=True)
        layers = []
        for j in range(len(ffn_sizes)-1):
            layers.append(nn.Linear(ffn_sizes[j], ffn_sizes[j+1]))
            if j<(len(ffn_sizes)-2):
                layers.append(activation())
            else:
                layers.append(output_activation())

        self.ffn = nn.Sequential(*layers)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad=False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad=True
            
    def forward(self, *args):
        """Forward method 
        
        Parameters
        ----------
        x : torch.Tensor
            Sequential input. Tensor of size (N,L,d) where N is batch size, L is lenght of the sequence, and d is dimension of the path
        Returns
        -------
        output : torch.Tensor
            Sequential output. Tensor of size (N, L, d_out) containing the output from the last layer of the RNN for each timestep
        """
        x = torch.cat(args, -1)
        output_RNN, _ = self.rnn(x)
        output = self.ffn(output_RNN)
        return output


class RNN_Taylor(nn.Module):

    def __init__(self, ffn_sizes, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()
        layers = []
        for j in range(len(ffn_sizes)-1):
            layers.append(nn.Linear(ffn_sizes[j], ffn_sizes[j+1]))
            if j<(len(ffn_sizes)-2):
                layers.append(activation())
            else:
                layers.append(output_activation())

        self.ffn = nn.Sequential(*layers)
        self.ffn_init = nn.Sequential(nn.Linear(ffn_sizes[0], 20), nn.ReLU(), nn.Linear(20, ffn_sizes[-1])) # init vvalue is not necessarily fixed, because it is a parametric PPDE

    def forward(self, *args):
        x = torch.cat(args, -1)
        assert x.dim()==3, "x needs to be a trajectory of shape (batch_size, L, dim)"
        batch_size = x.shape[0]
        device = x.device

        y = self.ffn(x[:,1:,:])
        y0 = self.ffn_init(x[:,0,:])
        y = torch.cat([y0.unsqueeze(1), y], 1)
        return y.cumsum(1) # Stochastic Taylor expansion assuming the input are the iterated integrals


