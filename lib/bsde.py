import torch
import torch.nn as nn
import signatory
from typing import Tuple, Optional, List
from abc import abstractmethod
import math

from lib.networks import RNN
from lib.options import Lookback, BaseOption, EuropeanCall
from lib.augmentations import *
from lib.augmentations import apply_augmentations



class PPDE(nn.Module):

    def __init__(self, d: int, mu: float, depth: int, rnn_hidden: int, ffn_hidden: List[int], continuous_approx: bool = True, **kwargs):
        """
        Base class that solves the parametric linear PPDE
        
        Parameters
        ----------
        d: int
            dimension of the PPDE
        mu: float
            risk-free rate
        depth: int
            depth of the signature
        rnn_hidden: int
            number of neurons in hidden layer of LSTM
        ffn_hidden: List[int]
            list with number of neurons per hidden layer in ffn after LSTM
        kwargs: dict()
            Additional keywords giving the dimensional of the additional parameters of the parametric PPDE. For example, one can have dim_sigma = 1


        """
        
        super().__init__()

        self.d = d
        dim_params_ppde = sum(kwargs.values()) if kwargs else 0
            
        self.mu = mu # risk free rate
        
        self.depth = depth
        self.augmentations = (AddTime(),)#(LeadLag(with_time=False),)
        #self.sig_channels = signatory.signature_channels(channels=2*d, depth=depth) # x2 because we do lead-lag
        self.sig_channels = signatory.signature_channels(channels=d+1, depth=depth) # +1 because we augment the path with time
        self.continuous_approx = continuous_approx
        if continuous_approx:
            self.f = RNN(rnn_in=self.sig_channels + 1 + dim_params_ppde, rnn_hidden=rnn_hidden, ffn_sizes=ffn_hidden+[1]) # +1 is for time
            self.dfdx = RNN(rnn_in=self.sig_channels + 1 + dim_params_ppde, rnn_hidden=rnn_hidden, ffn_sizes=ffn_hidden+[d]) # +1 is for time
        else:
            self.f = RNN(rnn_in=self.d + 1 + dim_params_ppde, rnn_hidden=rnn_hidden, ffn_sizes=ffn_hidden+[1]) # +1 is for time
            self.dfdx = RNN(rnn_in=self.d + 1 + dim_params_ppde, rnn_hidden=rnn_hidden, ffn_sizes=ffn_hidden+[d]) # +1 is for time


    @abstractmethod
    def sdeint(self, ts, x0, **kwargs):
        """
        Code here the SDE that the underlying assets follow
        """
        ...


    def prepare_data(self, ts: torch.Tensor, x0: torch.Tensor, lag: int, **kwargs):
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
        input_nn: torch.Tensor
            Discrete path, of path of signatures depending on the input of the NN
        sum_increments: torch.Tensor
            Increments of the Brownian motion on the coarse time discretisation. Tensor of shape (batch_size, n_steps/lag+1, d)
        """

        if self.continuous_approx:
            x, input_nn, sum_increments = self._prepare_data_with_signature(ts=ts, x0=x0, lag=lag, **kwargs)
        else:
            x, input_nn, sum_increments = self._prepare_data_without_signature(ts=ts, x0=x0, lag=lag, **kwargs)
        
        return x, input_nn, sum_increments

    def _prepare_data_without_signature(self, ts: torch.Tensor, x0: torch.Tensor, lag: int, **kwargs):
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
        input_nn: torch.Tensor
            Discrete path
        sum_increments: torch.Tensor
            Increments of the Brownian motion on the coarse time discretisation. Tensor of shape (batch_size, n_steps/lag+1, d)
        """
        x, brownian_increments = self.sdeint(ts, x0, **kwargs)
        device = x.device
        batch_size = x.shape[0]
        sum_increments = torch.zeros(batch_size, len(range(0, len(ts), lag)), self.d, device=device)

        for idx, id_t in enumerate(range(0, len(ts), lag)):
            try:
                sum_increments[:,idx,:] = torch.sum(brownian_increments[:,id_t:id_t+lag], 1)
            except:
                pass # it is the last point and we don't have anymore increments, but that's okay, because at the last step of the bsde, we compare thes olution of the bsde against the payoff of the option
        return x, x[:,::lag,:], sum_increments 
    
    
    def _prepare_data_with_signature(self, ts: torch.Tensor, x0: torch.Tensor, lag: int, **kwargs):
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
        x, brownian_increments = self.sdeint(ts, x0, **kwargs)
        device = x.device
        batch_size = x.shape[0]
        path_signature = torch.zeros(batch_size, len(range(0, len(ts), lag)), self.sig_channels, device=device)
        sum_increments = torch.zeros(batch_size, len(range(0, len(ts), lag)), self.d, device=device)
        basepoint = torch.zeros(batch_size, 1, self.d, device=device)

        for idx, id_t in enumerate(range(0, len(ts), lag)):
            if idx == 0:
                t = ts[:1].repeat(2)
                portion_path = torch.cat([basepoint, x[:,0,:].unsqueeze(1)],1)
            else:
                t = ts[id_t-lag:id_t+1]
                portion_path = x[:,id_t-lag:id_t+1,:] 
                
            augmented_path = apply_augmentations(portion_path, self.augmentations, AddTime=t)
            if self.continuous_approx:
                path_signature[:,idx,:] = signatory.signature(augmented_path, self.depth)
            try:
                sum_increments[:,idx,:] = torch.sum(brownian_increments[:,id_t:id_t+lag], 1)
            except:
                pass # it is the last point and we don't have anymore increments, but that's okay, because at the last step of the bsde, we compare thes olution of the bsde against the payoff of the option
        return x, path_signature, sum_increments 
    
    def get_stream_signatures(self, ts: torch.Tensor, x: torch.Tensor, lag: int):
        """
        Given a path, get the stream of signatures
        
        Parameters
        ----------
        ts: torch.Tensor
            Time discretisation.
        x: torch.Tensor
            Tensor of size (batch_size, n_steps, d)
        """
        device = x.device
        batch_size = x.shape[0]
        path_signature = torch.zeros(batch_size, len(range(0, len(ts), lag)), self.sig_channels, device=device)
        basepoint = torch.zeros(batch_size, 1, self.d, device=device)

        for idx, id_t in enumerate(range(0, len(ts), lag)):
            if idx == 0:
                t = ts[:1].repeat(2)
                portion_path = torch.cat([basepoint, x[:,0,:].unsqueeze(1)],1)
            else:
                t = ts[id_t-lag:id_t+1]
                portion_path = x[:,id_t-lag:id_t+1,:] 
            augmented_path = apply_augmentations(portion_path, self.augmentations, AddTime=t)
            path_signature[:,idx,:] = signatory.signature(augmented_path, self.depth)
        return path_signature
    
    def eval(self, ts: torch.Tensor, x: torch.Tensor, lag: int, **kwargs):
        """
        Calculate the approximation of the solution of the PPDE at (t,x)
        """
        x = x.unsqueeze(0) if x.dim()==2 else x
        device = x.device
        batch_size, id_t = x.shape[0], x.shape[1]
        if self.continuous_approx:
            input_nn = self.get_stream_signatures(ts=ts[:id_t], x=x, lag=lag)
        else:
            input_nn = x[:,::lag,:]

        t = ts[:id_t:lag].reshape(1,-1,1).repeat(batch_size,1,1)
        tx = torch.cat([t,input_nn],2)
        args = []
        for key in kwargs.keys():
            args.append(torch.ones_like(t) * kwargs[key])
        with torch.no_grad():
            Y = self.f(tx, *args) # (batch_size, L, 1)
        return Y[:,-1,:] # (batch_size, 1)
    
    def eval_mc(self, ts: torch.Tensor, x: torch.Tensor, lag: int, option: BaseOption, mc_samples: int, **kwargs):
        """
        Calculate the approximation of the solution of the PPDE using Monte Carlo
        """
        x = x.unsqueeze(0) if x.dim()==2 else x
        batch_size, id_t = x.shape[0], x.shape[1]
        x = torch.repeat_interleave(x, mc_samples, dim=0)
        device = x.device
        mc_paths, _ = self.sdeint(ts = ts[id_t-1:], x0 = x[:,-1,:], **kwargs) 
        x = torch.cat([x, mc_paths[:,1:,:]],1)
        payoff = torch.exp(-self.mu * (ts[-1]-ts[id_t-1])) * option.payoff(x)
        payoff = payoff.reshape(batch_size, mc_samples, 1).mean(1)
        return payoff

    
    def fbsdeint(self, ts: torch.Tensor, x0: torch.Tensor, option: BaseOption, lag: int, **kwargs): 
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

        x, path_signature, brownian_increments = self.prepare_data(ts,x0,lag, **kwargs)
        payoff = option.payoff(x) # (batch_size, 1)
        device = x.device
        batch_size = x.shape[0]
        t = ts[::lag].reshape(1,-1,1).repeat(batch_size,1,1)
        tx = torch.cat([t,path_signature],2)
        
        Y = self.f(tx) # (batch_size, L, 1)
        Z = self.dfdx(tx) # (batch_size, L, dim)

        loss_fn = nn.MSELoss()
        loss = 0
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




class PPDE_BlackScholes(PPDE):

    def __init__(self, d: int, mu: float, sigma: float, depth: int, rnn_hidden: int, ffn_hidden: List[int]):
        super(PPDE_BlackScholes, self).__init__(d=d, mu=mu, depth=depth, rnn_hidden=rnn_hidden, ffn_hidden=ffn_hidden)
        self.sigma = sigma # 
    
    
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

    

class PPDE_Heston(PPDE):

    def __init__(self, d: int, mu: float, vol_of_vol: float, kappa: float, theta: float,  depth: int, rnn_hidden: int, ffn_hidden: List[int]):
        assert d==2, "we need d=2"
        assert 2*kappa*theta > vol_of_vol , "Feller condition is not satisfied"
        super(PPDE_Heston, self).__init__(d=d, mu=mu, depth=depth, rnn_hidden=rnn_hidden, ffn_hidden=ffn_hidden)
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

class PPDE_RoughVol(PPDE):

    def __init__(self, mu: float, depth: int, rnn_hidden: int, ffn_hidden: List[int], kappa: float, V_infty: float, eta: float, H: float, rho: float, **kwargs):
        """
        Parametric PPDE to price options under Rough Volatility model

        Parameters
        ----------
        mu: float
            Risk-free rate
        depth: int
            Depth of signature
        rnn_hidden: int
            Number of neurons in hidden layer of LSTM
        ffn_hidden: int
            Number of neurons in hidden layer of output of LSTM
        kappa: float
            coefficient drift of vol process. See https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3400035
        V_infty: float
            target vol in the mean reverse process
        
        **kwargs
            additional dims of parameters in the parametric PPDE
            
        """

        super(PPDE_RoughVol, self).__init__(d=2, mu=mu, depth=depth, rnn_hidden=rnn_hidden, ffn_hidden=ffn_hidden, dim_strike=1)

        self.kappa=kappa
        self.V_infty = V_infty
        self.eta = eta
        self.H = H
        self.rho = rho

    def _K(self, t):
        """
        Kernel function in vol process
        """
        return t**(self.H-0.5) / math.gamma(self.H + 0.5)

    def sdeint(self, ts, x0):
        """
        Euler scheme to solve the SDE.
        Parameters
        ----------
        ts: troch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        """
        x = x0.unsqueeze(1)
        batch_size = x.shape[0]
        device = x.device
        brownian_increments = torch.randn(batch_size, len(ts)-1, self.d, device=device)
        brownian_increments[...,0] = self.rho * brownian_increments[...,0] + (1-self.rho) * brownian_increments[...,-1]

        
        for idx, t in enumerate(ts[1:]):
            h = ts[idx+1] - ts[idx]
            brownian_increments[:,idx,:] *= torch.sqrt(h)
            s_new = x[:,-1,0] + self.mu*x[:,-1,0]*h + x[:,-1,0]*torch.exp(x[:,-1,1])*brownian_increments[:,idx,0]

            driftV, diffV = 0, 0
            K = [self._K(ts[idx+1] - r) for r in ts[:idx+1]]
            for idt in range(idx+1):
                driftV += K[idt] * self.kappa * (x[:,idt,1] - self.V_infty) * h
                diffV += K[idt] * self.eta * x[:,idt,1] * brownian_increments[:,idt,1]
            v_new = x[:,0,1] + driftV + diffV

            x_new = torch.stack([s_new, v_new],1)
            x = torch.cat([x,x_new.unsqueeze(1)],1)
        return x, brownian_increments

    def fbsdeint_parametric(self, ts: torch.Tensor, x0: torch.Tensor, lag: int, **kwargs): 
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

        x, input_nn, brownian_increments = self.prepare_data(ts,x0,lag)
        device = x.device
        if kwargs.get('K') is not None:
            strikes = torch.ones(x.shape[0], device=device)*kwargs.get('K')
        else:
            strikes = 0.9 + 0.2* torch.rand(x.shape[0], device=device)
        option = EuropeanCall(strikes)
        payoff = option.payoff(x[:,-1,:]) # (batch_size, 1)
        batch_size = x.shape[0]
        t = ts[::lag].reshape(1,-1,1).repeat(batch_size,1,1)
        L = input_nn.shape[1]
        strikes = torch.repeat_interleave(strikes.reshape(-1,1,1), L, dim=1)
        
        Y = self.f(t, input_nn, strikes) # (batch_size, L, 1)
        Z = self.dfdx(t, input_nn, strikes) # (batch_size, L, dim)

        loss_fn = nn.MSELoss()
        loss = 0
        h = t[:,1:,:] - t[:,:-1,:] 
        discount_factor = torch.exp(-self.mu*h) # 
        target = discount_factor*Y[:,1:,:].detach()
        stoch_int = torch.sum(Z*brownian_increments,2,keepdim=True) # (batch_size, L, 1)
        pred = Y[:,:-1,:] + stoch_int[:,:-1,:] # (batch_size, L-1, 1)
        
        loss = torch.mean((pred-target)**2,0).sum()
        loss += loss_fn(Y[:,-1,:], payoff)
        
        return loss, Y, payoff
