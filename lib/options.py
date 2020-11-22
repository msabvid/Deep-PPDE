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
        payoff = torch.max(x, 1)[0]-x[:,-1,:] # (batch_size, d)
        return torch.sum(payoff, 1, keepdim=True) # (batch_size, 1)


