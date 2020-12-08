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
