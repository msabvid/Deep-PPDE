import torch
import torch.nn as nn
import numpy as np
import argparse
import tqdm
import os
import math
import pandas as pd
import matplotlib.pyplot as plt

from lib.bsde import PPDE_RoughVol as PPDE
from lib.options import EuropeanCall


def sample_x0(batch_size, device):
    sigma = 0.3
    mu = 0.08
    tau = 0.1
    z = torch.randn(batch_size, 1, device=device)
    s0 = torch.exp((mu-0.5*sigma**2)*tau + 0.3*math.sqrt(tau)*z) # lognormal
    s0 = torch.ones(batch_size,1, device=device)
    v0 = torch.ones_like(s0) * 0.04
    x0 = torch.cat([s0,v0],1)
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
        kappa,
        eta,
        V_infty,
        rho,
        H,
        depth,
        rnn_hidden,
        ffn_hidden,
        max_updates,
        batch_size, 
        lag,
        base_dir,
        device,
        method,
        continuous
        ):
    
    logfile = os.path.join(base_dir, "log.txt")
    ts = torch.linspace(0,T,n_steps+1, device=device)
    ppde = PPDE(mu=mu, kappa=kappa, V_infty=V_infty, eta=eta, rho=rho, H=H,
            depth=depth, rnn_hidden=rnn_hidden, ffn_hidden=ffn_hidden,
            continuous_approx=continuous)
    ppde.to(device)
    optimizer = torch.optim.RMSprop(ppde.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,2000], gamma=0.5)
    
    pbar = tqdm.tqdm(total=max_updates)
    losses = []
    for idx in range(max_updates):
        optimizer.zero_grad()
        x0 = sample_x0(batch_size, device)
        if method=="bsde":
            loss, _, _ = ppde.fbsdeint_parametric(ts=ts, x0=x0, lag=lag)
        #else:
        #    loss, _, _ = ppde.conditional_expectation(ts=ts, x0=x0, option=lookback, lag=lag)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().item())
        # testing
        if (idx+1)%20 == 0:
            with torch.no_grad():
                x0 = torch.ones(5000,d,device=device) # we do monte carlo
                x0[:,1] = x0[:,1]*0.04
                loss, Y, payoff = ppde.fbsdeint_parametric(ts=ts,x0=x0,lag=lag, K=1)
                payoff = torch.exp(-mu*ts[-1])*payoff.mean()
                write("loss={:.4f}, Monte Carlo price={:.4f}, predicted={:.4f}".format(loss.item(),payoff.item(), Y[0,0,0].item()),logfile,pbar)
        pbar.update(1)
    # save results
    result = {"f":ppde.f.state_dict(),
              "dfdx":ppde.dfdx.state_dict(),
              "loss":losses}
    torch.save(result, os.path.join(base_dir, "result.pth.tar"))
    # make plots
    evaluate(**locals())


def evaluate(T,
        n_steps,
        d,
        mu,
        kappa,
        eta,
        V_infty,
        rho,
        H,
        depth,
        rnn_hidden,
        ffn_hidden,
        lag,
        base_dir,
        device,
        continuous,
        **kwargs
        ):
    
    logfile = os.path.join(base_dir, "log.txt")
    ts = torch.linspace(0,T,n_steps+1, device=device)
    ppde = PPDE(mu=mu, kappa=kappa, V_infty=V_infty, eta=eta, rho=rho, H=H,
            depth=depth, rnn_hidden=rnn_hidden, ffn_hidden=ffn_hidden,
            continuous_approx=continuous)
    ppde.to(device)
    state = torch.load(os.path.join(base_dir, "result.pth.tar"), map_location=device)
    ppde.f.load_state_dict(state["f"])
    ppde.dfdx.load_state_dict(state["dfdx"])
    
    # make plots
    # 1) Option price at t=0
    price_mc, price_pred = [], []
    for K in np.linspace(0.9,1.1,11):
        with torch.no_grad():
            x0=torch.ones(10000,2,device=device)
            x0[:,1] = x0[:,1]*0.04
            loss, Y, payoff = ppde.fbsdeint_parametric(ts=ts,x0=x0,lag=lag,K=K)
        payoff = torch.exp(-mu*ts[-1])*payoff.mean()
        price_mc.append(payoff.item())
        price_pred.append(Y[0,0,0].item())
    df = pd.DataFrame({'K':np.linspace(0.9,1.1,11),
                       'price_mc':price_mc,
                       'price_pred':price_pred})
    df.to_csv(os.path.join(base_dir, "df.csv"))
    
    # 2) approximation of PPDE solution along a path
    for paths in range(10):
        print("path {}".format(paths))
        x0 = torch.ones(1,d,device=device)#sample_x0(1, d, device)
        x0[:,1] = x0[:,1]*0.04
        with torch.no_grad():
            x, _ = ppde.sdeint(ts=ts, x0=x0)
        
        fig = plt.figure(figsize=(12,5))
        spec = fig.add_gridspec(2,2)
        
        ax0 = fig.add_subplot(spec[:,0])
        ax0.plot(ts.cpu().numpy(), x[0,:,0].cpu().numpy())
        ax0.set_ylabel(r"$S(t)$")
        
        for idlag, lag_eval in enumerate([lag//2, lag]):
            price_pred, price_mc = [], []
            for idx, t in enumerate(ts[::lag_eval]):
                price_pred.append(ppde.eval(ts=ts, x=x[:,:(idx*lag_eval)+1,:], lag=lag_eval, K=1).detach())
                option = EuropeanCall(K=1)
                price_mc.append(ppde.eval_mc(ts=ts, x=x[:,:(idx*lag_eval)+1,:], lag=lag_eval, option=option, mc_samples=10000))
            price_pred = torch.cat(price_pred, 0).view(-1).cpu().numpy()
            price_mc = torch.cat(price_mc, 0).view(-1).cpu().numpy()
            
            ax = fig.add_subplot(spec[idlag, 1])
            ax.plot(ts[::lag_eval].cpu().numpy(), price_pred, '--', label="Deep Learning price")
            ax.plot(ts[::lag_eval].cpu().numpy(), price_mc, '-', label="Monte Carlo price")
            ax.set_ylabel(r"$v(t,X_t)$")
            ax.set_title("Lag {}".format(lag_eval))
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(base_dir, "RoughVol_path{}.pdf".format(paths)))












if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', default='./numerical_results/', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--d', default=2, type=int)
    parser.add_argument('--max_updates', default=5000, type=int)
    parser.add_argument('--ffn_hidden', default=[20,10], nargs="+", type=int)
    parser.add_argument('--rnn_hidden', default=20, type=int)
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--n_steps', default=100, type=int, help="number of steps in time discrretisation")
    parser.add_argument('--lag', default=10, type=int, help="lag in fine time discretisation to create coarse time discretisation")
    parser.add_argument('--mu', default=0.05, type=float, help="risk free rate")
    parser.add_argument('--kappa', default=0.5, type=float, help="mean reverting process coef")
    parser.add_argument('--V_infty', default=0.1, type=float, help="target variance")
    parser.add_argument('--eta', default=0.8, type=float, help="coef diff vol")
    parser.add_argument('--H', default=0.25, type=float, help="Hurst parameter")
    parser.add_argument('--rho', default=0., type=float, help="correlation brownian motions")
    parser.add_argument('--method', default="bsde", type=str, help="learning method", choices=["bsde","orthogonal"])
    
    parser.add_argument('--continuous', default=False, action='store_true')
    parser.add_argument('--evaluate', default=False, action='store_true')
    

    args = parser.parse_args()
    
    assert args.d==2, "Heston implementation is for d=2" 
    
    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda:{}".format(args.device)
    else:
        device="cpu"
    
    if args.continuous:
        results_path = os.path.join(args.base_dir, "RoughVol", args.method, 'signature')
    else:
        results_path = os.path.join(args.base_dir, "RoughVol", args.method, 'discrete')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    
    if args.evaluate:
        evaluate(T=args.T,
            n_steps=args.n_steps,
            d=args.d,
            mu=args.mu,
            kappa=args.kappa,
            V_infty=args.V_infty,
            eta=args.eta,
            rho=args.rho,
            H=args.H,
            depth=args.depth,
            rnn_hidden=args.rnn_hidden,
            ffn_hidden=args.ffn_hidden,
            max_updates=args.max_updates,
            batch_size=args.batch_size,
            lag=args.lag,
            base_dir=results_path,
            device=device,
            method=args.method,
            continuous=args.continuous
            )
    else:
        train(T=args.T,
            n_steps=args.n_steps,
            d=args.d,
            mu=args.mu,
            kappa=args.kappa,
            V_infty=args.V_infty,
            eta=args.eta,
            rho=args.rho,
            H=args.H,
            depth=args.depth,
            rnn_hidden=args.rnn_hidden,
            ffn_hidden=args.ffn_hidden,
            max_updates=args.max_updates,
            batch_size=args.batch_size,
            lag=args.lag,
            base_dir=results_path,
            device=device,
            method=args.method,
            continuous=args.continuous
            )
