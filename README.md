# SGU-MLP

Pytorch implementation of the SGU-MLP Architecture (mostly) as described in the paper "Spatial Gated Multi-Layer Perceptron for Land Use".
This implementation adds configurable and learnable residual weights for the input processing. 
The rest is following the [implementation of the original authors](https://github.com/aj1365/SGUMLP/blob/main/SGUMLP.ipynb). 

***Note:*** This means the implementation differs from the architecture described in the paper in the following aspects:

- Input patches and DWC Block outputs are combined using a residual connection.
- Tokens are embedded using a Convolutional Layer, not projected pixel-wise (to project patches pixelwise, set `embedding_kernel_size=1`)
- Input patches are overlapping, not non-overlapping (this is only relevant for data preprocessing).

## Installation

If you just want to use SGU-MLP:
````bash
pip install git+https://github.com/simulacrum6/sgu-mlp.git
````

If you want to run experiments as well:
````bash
pip install "git+https://github.com/simulacrum6/sgu-mlp.git#egg=sgu-mlp[experiments]"
````


## Running Experiments

To run the replication experiment, first download the benchmark datasets. 
You can download them from [gDrive](https://drive.usercontent.google.com/download?id=1dLJJrNJpQoQeDHybs37iGxmrSU6aP2xv&export=download) or run

````bash
python3 -m experiments.run download --out_dir='/path/for/data/'
````

To run the replication experiment:
````bash
python3 -m experiments.run experiment replication
````
Per default, it is assumed, that you run the script from the root of this repository, so ``--root_dir='data/config``. 
Adjust this to suit your needs.  

To run an arbitrary experiment:
````bash
python3 -m experiments.run run <experiment_type> <cfg_path>
````
**Arguments**
- `<experiment_type`: "cv" or "ood".
- `<cfg_path>`: path to the config file for the experiment.

See `data/config` for examples.

# References
- _[Spatial Gated Multi-Layer Perceptron for Land Use and Land Cover Mapping](https://github.com/aj1365/SGUMLP/blob/main/Spatial_Gated_Multi-Layer_Perceptron_for_Land_Use_and_Land_Cover_Mapping.pdf)_ - Jamali et al. 2024
- _[Pay Attention to MLPs](https://dl.acm.org/doi/10.5555/3540261.3540965)_ - Liu et al. 2021
- [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601) - Tolstikhin et al. 2021