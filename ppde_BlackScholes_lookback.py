import torch
import torch.nn as nn
import numpy as np
import argparse
import tqdm
import os
import math
import matplotlib.pyplot as plt

from lib.bsde import PPDE_BlackScholes as PPDE
from lib.options import Lookback


def sample_x0(batch_size, dim, device):
    sigma = 0.3
    mu = 0.08
    tau = 0.1
    z = torch.randn(batch_size, dim, device=device)
    x0 = torch.exp((mu-0.5*sigma**2)*tau + 0.3*math.sqrt(tau)*z) # lognormal
    return x0
    

def write(msg, logfile, pbar):
    pbar.write(msg)
    with open(logfile, "a") as f:
        f.write(msg)
        f.write("\n")


def train(T,
        n_steps,
        d,
        mu,
        sigma,
        depth,
        rnn_hidden,
        ffn_hidden,
        max_updates,
        batch_size, 
        lag,
        base_dir,
        device,
        method
        ):
    
    logfile = os.path.join(base_dir, "log.txt")
    ts = torch.linspace(0,T,n_steps+1, device=device)
    lookback = Lookback()
    ppde = PPDE(d, mu, sigma, depth, rnn_hidden, ffn_hidden)
    ppde.to(device)
    optimizer = torch.optim.RMSprop(ppde.parameters(), lr=0.0005)
    
    pbar = tqdm.tqdm(total=max_updates)
    losses = []
    for idx in range(max_updates):
        optimizer.zero_grad()
        x0 = sample_x0(batch_size, d, device)
        if method=="bsde":
            loss, _, _ = ppde.fbsdeint(ts=ts, x0=x0, option=lookback, lag=lag)
        else:
            loss, _, _ = ppde.conditional_expectation(ts=ts, x0=x0, option=lookback, lag=lag)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())
        # testing
        if (idx+1) % 10 == 0:
            with torch.no_grad():
                x0 = torch.ones(5000,d,device=device) # we do monte carlo
                loss, Y, payoff = ppde.fbsdeint(ts=ts,x0=x0,option=lookback,lag=lag)
                payoff = torch.exp(-mu*ts[-1])*payoff.mean()
            
            pbar.update(10)
            write("loss={:.4f}, Monte Carlo price={:.4f}, predicted={:.4f}".format(loss.item(),payoff.item(), Y[0,0,0].item()),logfile,pbar)
    
    result = {"state":ppde.state_dict(),
            "loss":losses}
    torch.save(result, os.path.join(base_dir, "result.pth.tar"))


    # evaluation
    x0 = torch.ones(1,d,device=device)#sample_x0(1, d, device)
    with torch.no_grad():
        x, _ = ppde.sdeint(ts=ts, x0=x0)
    fig, ax = plt.subplots()
    ax.plot(x[0,:,0].cpu().numpy())
    fig.savefig(os.path.join(base_dir, "path_eval.pdf"))
    pred, mc_pred = [], []
    for idx, t in enumerate(ts[::lag]):
        pred.append(ppde.eval(ts=ts, x=x[:,:(idx*lag)+1,:], lag=lag).detach())
        mc_pred.append(ppde.eval_mc(ts=ts, x=x[:,:(idx*lag)+1,:], lag=lag, option=lookback, mc_samples=10000))
    print("THE END")






if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', default='./numerical_results/', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--d', default=4, type=int)
    parser.add_argument('--max_updates', default=5000, type=int)
    parser.add_argument('--ffn_hidden', default=[20,20], nargs="+", type=int)
    parser.add_argument('--rnn_hidden', default=20, type=int)
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--T', default=1., type=float)
    parser.add_argument('--n_steps', default=100, type=int, help="number of steps in time discrretisation")
    parser.add_argument('--lag', default=10, type=int, help="lag in fine time discretisation to create coarse time discretisation")
    parser.add_argument('--mu', default=0.05, type=float, help="risk free rate")
    parser.add_argument('--sigma', default=0.3, type=float, help="risk free rate")
    parser.add_argument('--method', default="bsde", type=str, help="learning method", choices=["bsde","orthogonal"])
    

    args = parser.parse_args()
    
    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda:{}".format(args.device)
    else:
        device="cpu"
    
    results_path = os.path.join(args.base_dir, "BS", args.method)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    train(T=args.T,
        n_steps=args.n_steps,
        d=args.d,
        mu=args.mu,
        sigma=args.sigma,
        depth=args.depth,
        rnn_hidden=args.rnn_hidden,
        ffn_hidden=args.ffn_hidden,
        max_updates=args.max_updates,
        batch_size=args.batch_size,
        lag=args.lag,
        base_dir=results_path,
        device=device,
        method=args.method
        )
