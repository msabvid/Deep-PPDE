import torch
from dataclasses import dataclass
from abc import abstractmethod

@dataclass
class BaseOption:
    pass
    
    @abstractmethod
    def payoff(self, x: torch.Tensor, **kwargs):
        ...



@dataclass
class Lookback(BaseOption):
    pass
    
    def payoff(self, x, **kwargs):
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
        basket = torch.sum(x,2) # (batch_size, N)
        payoff = torch.max(basket, 1)[0]-basket[:,-1] # (batch_size)
        return payoff.unsqueeze(1) # (batch_size, 1)


