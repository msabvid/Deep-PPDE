# On solving path dependent PDEs (PPDE) / Pricing path-dependent derivatives
Code to solve PPDEs. For more details, go [here](https://arxiv.org/abs/2011.10630)
    
    @misc{sabatevidales2020solving,
          title={Solving path dependent PDEs with LSTM networks and path signatures}, 
          author={Marc Sabate-Vidales and David Šiška and Lukasz Szpruch},
          year={2020},
          eprint={2011.10630},
          archivePrefix={arXiv},
          primaryClass={q-fin.CP}
    }


We use LSTM networks as non-anticipative functionals, and path signatures to price path dependent derivatives at any time t, given the asset price history.
This is equivalent to solving
![](/images_readme/ppde.png)

We use two different learning algorithms:
- The BSDE method --> we also learn the hedging strategy.
- Learning the conditional expectation E[X|F] as the orthogonal projection of X onto the sigma-algebra F. 

## Running the code

```
usage: ppde_BlackScholes_lookback.py [-h] [--base_dir BASE_DIR] [--device DEVICE] [--use_cuda] [--seed SEED] [--batch_size BATCH_SIZE] [--d D] [--max_updates MAX_UPDATES]
                                     [--ffn_hidden FFN_HIDDEN [FFN_HIDDEN ...]] [--rnn_hidden RNN_HIDDEN] [--depth DEPTH] [--T T] [--n_steps N_STEPS] [--lag LAG] [--mu MU] [--sigma SIGMA]
                                     [--method {bsde,orthogonal}]

optional arguments:
  -h, --help            show this help message and exit
  --base_dir BASE_DIR
  --device DEVICE
  --use_cuda
  --seed SEED
  --batch_size BATCH_SIZE
  --d D
  --max_updates MAX_UPDATES
  --ffn_hidden FFN_HIDDEN [FFN_HIDDEN ...]
  --rnn_hidden RNN_HIDDEN
  --depth DEPTH
  --T T
  --n_steps N_STEPS     number of steps in time discrretisation
  --lag LAG             lag in fine time discretisation to create coarse time discretisation
  --mu MU               risk free rate
  --sigma SIGMA         volatility
  --method {bsde,orthogonal}
                        learning method
```

For example, training the network using the BSDE method:
```
python ppde_BlackScholes_lookback.py --use_cuda --device 0 --batch_size 2000 --max_updates 20000 --T 0.5 --method bsde
```

## Acknowledgements
Path augmentations code taken from https://github.com/SigCGANs/Conditional-Sig-Wasserstein-GANs


## Example
Solving the Black-Scholes PPDE to price the Lookback option.
```
import torch
import torch.nn as nn
import numpy as np
import argparse
import tqdm
import os
import math
import matplotlib.pyplot as plt

from lib.bsde import PPDE_BlackScholes as PPDE # PPDE solver
from lib.options import Lookback
```
We generate paths to train using the associated SDE. The initial point of the paths is sampled from a lognormal distribution
```
def sample_x0(batch_size, dim, device):
    sigma = 0.3
    mu = 0.08
    tau = 0.1
    z = torch.randn(batch_size, dim, device=device)
    x0 = torch.exp((mu-0.5*sigma**2)*tau + 0.3*math.sqrt(tau)*z) # lognormal
    return x0
```
We initialise the problem
```
T = 1. # terminal time
n_steps = 500 # number of steps in time discretisation where we solve the SDE
d = 1 # dimension of X
mu = 0.05 # risk-free rate
sigma = 0.3 # volatility 
depth = 3 # depth of signature
rnn_hidden = 20 # dimension of hidden layer of LSTM network
max_updates = 1000 # number of SGD steps
batch_size = 500
lag = 20 # we calculate the soluton of the PPDE every lag steps of the time discretisation
base_dir = './numerical_results'
device=0
method='bsde' # we us the bsde method to learn the solution of the PPDE
```
We set the cuda device, and the path where we save the results
```
if torch.cuda.is_available():
    device = "cuda:{}".format(args.device)
else:
    device="cpu"

results_path = os.path.join(base_dir, "BS", method)
if not os.path.exists(results_path):
os.makedirs(results_path)
```
We set the time discretisation
```
ts = torch.linspace(0,T,n_steps+1, device=device)  
```
We set the terminal condition
```
lookback = Lookback() # in order to calculate the payoff of a path x where x has shape (batch_size, L, d), do lookback.payoff(x)
```
We initialise the PPDE solver, and the optimizer
```
ppde = PPDE(d, mu, sigma, depth, rnn_hidden, ffn_hidden).to(device)
optimizer = torch.optim.RMSprop(ppde.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.2)
```
We train
```
pbar = tqdm.tqdm(max_updates)
for idx in range(max_updates):
    optimizer.zero_grad()
    x0 = sample_x0(batch_size, d, device)
    loss, _, _ = ppde.fbsdeint(ts=ts, x0=x0, option=lookback, lag=lag)
    loss.backward()
    optimizer.step()
    scheduler.step()
    pbar.update(1)
```
We plot the result. For this we generate one random path of the price of the underlying asset, and we evaluate the learned solution of the PPDE on this path, in each timestep of the time discretisation. We compare against the solution estimated using Monte Carlo. 
We generate the path and we plot it:
```
x0 = torch.ones(1,d,device=device)#sample_x0(1, d, device)
with torch.no_grad():
        x, _ = ppde.sdeint(ts=ts, x0=x0)
fig, ax = plt.subplots()
ax.plot(ts.cpu().numpy(), x[0,:,0].cpu().numpy())
ax.set_ylabel(r"$X(t)$")
fig.savefig(os.path.join(base_dir, "path_eval.pdf"))
pred, mc_pred = [], []
for idx, t in enumerate(ts[::lag]):
    pred.append(ppde.eval(ts=ts, x=x[:,:(idx*lag)+1,:], lag=lag).detach())
    mc_pred.append(ppde.eval_mc(ts=ts, x=x[:,:(idx*lag)+1,:], lag=lag, option=lookback, mc_samples=10000))
    pred = torch.cat(pred, 0).view(-1).cpu().numpy()
    mc_pred = torch.cat(mc_pred, 0).view(-1).cpu().numpy()
fig, ax = plt.subplots()
ax.plot(ts[::lag].cpu().numpy(), pred, '--', label="LSTM + BSDE + sign")
ax.plot(ts[::lag].cpu().numpy(), mc_pred, '-', label="MC")
ax.set_ylabel(r"$v(t,X_t)$")
ax.legend()
fig.savefig(os.path.join(base_dir, "BS_lookback_LSTM_sol.pdf"))
print("THE END")

```
![](/images_readme/path_eval.png)
![](/images_readme/Bs_lookback_LSTM_sol.png)
