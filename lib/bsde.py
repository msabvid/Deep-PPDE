import torch
import torch.nn as nn
import signatory
from typing import Tuple, Optional, List

from lib.networks import RNN
from lib.options import Lookback
from lib.augmentations import *
from lib.augmentations import apply_augmentations



class FBSDE(nn.Module):

    def __init__(self, ts: torch.Tensor, d: int, mu: int, sigma: int, depth: int, rnn_hidden: int, ffn_hidden: List[int]):
        super().__init__()
        self.d = d
        self.mu = mu
        self.sigma = sigma # change it to a parameter to solve a parametric family of PPDEs
    
        # deep learning approximations of ppde solution and ppde gradient
        self.depth = depth
        self.augmentations = (LeadLag(with_time=False),)
        self.sig_channels = signatory.signature_channels(channels=2*d, depth=depth) # x2 because we do lead-lag
        self.f = RNN(rnn_in=self.sig_channels+1, rnn_hidden=rnn_hidden, ffn_sizes=ffn_hidden+[1]) # +1 is for time
        self.dfdx = RNN(rnn_in=self.sig_channels+1, rnn_hidden=rnn_hidden, ffn_sizes=ffn_hidden+[d]) # +1 is for time

    def sdeint(self, ts, x0):
        """
        Parameters
        ----------
        ts: troch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        brownian: Optional. 
            torch.tensor of shape (batch_size, N, d)
        """
        x = x0.unsqueeze(1)
        batch_size = x.shape[0]
        device = x.device
        brownian_increments = torch.zeros(batch_size, len(ts)-1, self.d, device=device)
        for idx, t in enumerate(ts[1:]):
            h = ts[idx+1]-ts[idx]
            brownian_increments[:,idx,:] = torch.randn(batch_size, self.d, device=device)*torch.sqrt(h)
            x_new = x[:,-1,:] + self.mu*x[:,-1,:]*h + self.sigma*x[:,-1,:]*brownian_increments[:,idx,:]
            x = torch.cat([x, x_new.unsqueeze(1)],1)
        return x, brownian_increments

    
    def prepare_data(self, ts: torch.Tensor, x0: torch.Tensor, lag: int):
        x, brownian_increments = self.sdeint(ts, x0)
        device = x.device
        batch_size = x.shape[0]
        path_signature = torch.zeros(batch_size, len(range(0, len(ts), lag)), self.sig_channels, device=device)
        sum_increments = torch.zeros(batch_size, len(range(0, len(ts), lag)), self.d, device=device)
        basepoint = torch.zeros(batch_size, 1, self.d, device=device)

        for idx, id_t in enumerate(range(0, len(ts), lag)):
            if idx == 0:
                portion_path = torch.cat([basepoint, x[:,0,:].unsqueeze(1)],1)
            else:
                portion_path = x[:,id_t-lag:id_t+1,:] 
                
            augmented_path = apply_augmentations(portion_path, self.augmentations)
            path_signature[:,idx,:] = signatory.signature(augmented_path, self.depth)
            try:
                sum_increments[:,idx,:] = torch.sum(brownian_increments[:,id_t:id_t+lag], 1)
            except:
                pass # it is the last point and we don't have anymore increments, but that's okay, because at the last step of the bsde, we compare thes olution of the bsde against the payoff of the option
        return x, path_signature, sum_increments 

    
    def bsdeint(self, ts: torch.Tensor, x0: torch.Tensor, option: Lookback, lag: int): 
        """
        Parameters
        ----------
        ts: troch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        option: object of class option to calculate payoff
        lag: int
            lag in fine time discretisation
        
        """

        x, path_signature, brownian_increments = self.prepare_data(ts,x0,lag)
        payoff = option.payoff(x) # (batch_size, 1)
        device = x.device
        batch_size = x.shape[0]
        t = ts[::lag].reshape(1,-1,1).repeat(batch_size,1,1)
        tx = torch.cat([t,path_signature],2)
        
        Y = self.f(tx) # (batch_size, L, 1)
        Z = self.dfdx(tx) # (batch_size, L, dim)

        loss_fn = nn.MSELoss()
        loss = 0
        for idx,t in enumerate(ts[::lag]):
            discount_factor = torch.exp(-self.mu*(ts[idx+lag]-t))
            target = payoff if t==ts[-1] else discount_factor*Y[:,idx+1,:].detach()
            stoch_int = torch.sum(Z[:,idx,:]*brownian_increments[:,idx,:], 1, keepdim=True)
            pred = Y[:,idx,:] + stoch_int
            loss += loss_fn(pred, target)
        return loss, Y, payoff
            
            


            









        
