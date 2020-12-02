import torch
import torch.nn as nn
import signatory
from typing import Tuple, Optional, List
from abc import abstractmethod

from lib.networks import RNN
from lib.options import Lookback
from lib.augmentations import *
from lib.augmentations import apply_augmentations



class FBSDE(nn.Module):

    def __init__(self, d: int, mu: float, depth: int, rnn_hidden: int, ffn_hidden: List[int]):
        super().__init__()
        self.d = d
        self.mu = mu # risk free rate

        self.depth = depth
        self.augmentations = (LeadLag(with_time=False),)
        self.sig_channels = signatory.signature_channels(channels=2*d, depth=depth) # x2 because we do lead-lag
        self.f = RNN(rnn_in=self.sig_channels+1, rnn_hidden=rnn_hidden, ffn_sizes=ffn_hidden+[1]) # +1 is for time
        self.dfdx = RNN(rnn_in=self.sig_channels+1, rnn_hidden=rnn_hidden, ffn_sizes=ffn_hidden+[d]) # +1 is for time

    @abstractmethod
    def sdeint(self, ts, x0):
        """
        Code here the SDE that the underlying assets follow
        """
        ...

    
    def prepare_data(self, ts: torch.Tensor, x0: torch.Tensor, lag: int):
        """
        Prepare the data:
            1. Solve the sde using some sde solver on a fine time discretisation
            2. calculate path signature between consecutive timesteps of a coarser time discretisation
            3. Calculate increments of brownian motion on the coarser time discretisation
        Parameters
        ----------
        ts: torch.Tensor
            Time discrstisation. Tensor of size (n_steps + 1)
        x0: torch.Tensor
            initial value of paths. Tensor of size (batch_size, d)
        lag: int
            lag used to create the coarse time discretisation in terms of the fine time discretisation.
        
        Returns
        -------
        x: torch.Tensor
            Solution of the SDE on the fine time discretisation. Tensor of shape (batch_size, n_steps+1, d)
        path_signature: torch.Tensor
            Stream of signatures. Tensor of shape (batch_size, n_steps/lag + 1, sig_channels)
        sum_increments: torch.Tensor
            Increments of the Brownian motion on the coarse time discretisation. Tensor of shape (batch_size, n_steps/lag+1, d)
        """
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
        #for idx,t in enumerate(ts[::lag]):
        #    if t==ts[-1]:
        #        target = payoff
        #    else:
        #        discount_factor = torch.exp(-self.mu*(ts[idx*lag+lag]-t))
        #        target = discount_factor*Y[:,idx+1,:].detach()
        #    stoch_int = torch.sum(Z[:,idx,:]*brownian_increments[:,idx,:], 1, keepdim=True)
        #    pred = Y[:,idx,:] + stoch_int # if t==ts[-1], then it is already taken into account that stoch_int=0, because the increment of Brownian motion is 0, therefore we are indeed comparing against the payoff
        #    loss += loss_fn(pred, target)
        h = t[:,1:,:] - t[:,:-1,:] 
        discount_factor = torch.exp(-self.mu*h) # 
        target = discount_factor*Y[:,1:,:].detach()
        stoch_int = torch.sum(Z*brownian_increments,2,keepdim=True) # (batch_size, L, 1)
        pred = Y[:,:-1,:] + stoch_int[:,:-1,:] # (batch_size, L-1, 1)
        loss = torch.mean((pred-target)**2,0).sum()
        loss += loss_fn(Y[:,-1,:], payoff)
        return loss, Y, payoff
            
            
    def conditional_expectation(self, ts: torch.Tensor, x0: torch.Tensor, option: Lookback, lag: int): 
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

        loss_fn = nn.MSELoss()
        loss = 0
        for idx,t in enumerate(ts[::lag]):
            discount_factor = torch.exp(-self.mu*(ts[-1]-t))
            target = discount_factor*payoff 
            pred = Y[:,idx,:] 
            loss += loss_fn(pred, target)
        return loss, Y, payoff

    def unbiased_price(self, ts: torch.Tensor, x0:torch.Tensor, option: Lookback, lag: int, MC_samples: int):
        """
        We calculate an unbiased estimator of the price at time t=0 (for now) using Monte Carlo, and the stochastic integral as a control variate
        Parameters
        ----------
        ts: troch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (1, d)
        option: object of class option to calculate payoff
        lag: int
            lag in fine time discretisation
        MC_samples: int
            Monte Carlo samples
        """
        assert x0.shape[0] == 1, "we need just 1 sample"
        x0 = x0.repeat(MC_samples, 1)
        x, path_signature, brownian_increments = self.prepare_data(ts,x0,lag)
        payoff = option.payoff(x) # (batch_size, 1)
        device = x.device
        batch_size = x.shape[0]
        t = ts[::lag].reshape(1,-1,1).repeat(batch_size,1,1)
        tx = torch.cat([t,path_signature],2)
        
        with torch.no_grad():
            Z = self.dfdx(tx) # (batch_size, L, dim)
        stoch_int = 0
        for idx,t in enumerate(ts[::lag]):
            discount_factor = torch.exp(-self.mu *t)
            stoch_int += discount_factor * torch.sum(Z[:,idx,:]*brownian_increments[:,idx,:], 1, keepdim=True)
        
        return payoff, torch.exp(-self.mu*ts[-1])*payoff-stoch_int # stoch_int has expected value 0, thus it doesn't add any bias to the MC estimator, and it is correlated with payoff




class FBSDE_BlackScholes(FBSDE):

    def __init__(self, d: int, mu: float, sigma: float, depth: int, rnn_hidden: int, ffn_hidden: List[int]):
        super(FBSDE_BlackScholes, self).__init__(d=d, mu=mu, depth=depth, rnn_hidden=rnn_hidden, ffn_hidden=ffn_hidden)
        self.sigma = sigma # change it to a torch.parameter to solve a parametric family of PPDEs
    
    
    def sdeint(self, ts, x0):
        """
        Euler scheme to solve the SDE.
        Parameters
        ----------
        ts: troch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        brownian: Optional. 
            torch.tensor of shape (batch_size, N, d)
        Note
        ----
        I am assuming uncorrelated Brownian motion
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

    

class FBSDE_Heston(FBSDE):

    def __init__(self, d: int, mu: float, vol_of_vol: float, kappa: float, theta: float,  depth: int, rnn_hidden: int, ffn_hidden: List[int]):
        assert d==2, "we need d=2"
        assert 2*kappa*theta > vol_of_vol , "Feller condition is not satisfied"
        super(FBSDE_Heston, self).__init__(d=d, mu=mu, depth=depth, rnn_hidden=rnn_hidden, ffn_hidden=ffn_hidden)
        self.vol_of_vol = vol_of_vol
        self.kappa = kappa
        self.theta = theta
    
    
    def sdeint(self, ts, x0):
        """
        Euler scheme to solve the SDE.
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
            s_new = x[:,-1,0] + self.mu*x[:,-1,0]*h + x[:,-1,0]*torch.sqrt(x[:,-1,1])*brownian_increments[:,idx,0]
            v_new = x[:,-1,1] + self.kappa*(self.theta-x[:,-1,1])*h + self.vol_of_vol*torch.sqrt(x[:,-1,1])*brownian_increments[:,idx,1]
            x_new = torch.stack([s_new, v_new], 1) # (batch_size, 2)
            x = torch.cat([x, x_new.unsqueeze(1)],1)
        return x, brownian_increments
