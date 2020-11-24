import torch
from dataclasses import dataclass
from abc import abstractmethod
from typing import List

@dataclass
class BaseOption:
    pass
    
    @abstractmethod
    def payoff(self, x: torch.Tensor, **kwargs):
        ...



class Lookback(BaseOption):
    
    def __init__(self, idx_traded: List[int]=None):
        self.idx_traded = idx_traded # indices of traded assets. If None, then all assets are traded
    
    def payoff(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Path history. Tensor of shape (batch_size, N, d) where N is path length
        Returns
        -------
        payoff: torch.Tensor
            lookback option payoff. Tensor of shape (batch_size,1)
        """
        if self.idx_traded:
            basket = torch.sum(x[..., self.idx_traded],2) # (batch_size, N)
        else:
            basket = torch.sum(x,2) # (batch_size, N)
        payoff = torch.max(basket, 1)[0]-basket[:,-1] # (batch_size)
        return payoff.unsqueeze(1) # (batch_size, 1)


class Autocallable(BaseOption):
    
    def __init__(self, idx_traded: int, B: int, Q1: float, Q2: float, q: float, r: float, ts: torch.Tensor):
        """
        Autocallable option with 
        - two observation dates (T/3, 2T/3), 
        - premature payoffs Q1 and Q2
        - redemption payoff q*s
        """
        
        self.idx_traded = idx_traded # index of traded asset
        self.B = B # barrier
        self.Q1 = Q1
        self.Q2 = Q2
        self.q = q # redemption payoff
        self.r = r # risk-free rate
        self.ts = ts # timegrid
    
    def payoff(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Path history. Tensor of shape (batch_size, N, d) where N is path length
        Returns
        -------
        payoff: torch.Tensor
            autocallable option payoff. Tensor of shape (batch_size,1)
        """
        id_t1 = len(self.ts)//3
        mask1 = x[:, id_t1, self.idx_traded]>=self.B
        id_t2 = 2*len(self.ts)//3
        mask2 = x[:, id_t2, self.idx_traded]>=self.B

        payoff = mask1 * self.Q1 * torch.exp(self.r*(self.ts[-1]-self.ts[id_t1])) # we get the payoff Q1, and we put in a risk-less acount for the remaining time
        payoff += ~mask1 * mask2 * self.Q2 * torch.exp(self.r*(self.ts[-1]-self.ts[id_t2]))
        payoff += ~mask1 * (~mask2) * self.q*x[:,-1,self.idx_traded]

        return payoff.unsqueeze(1) # (batch_size, 1)


