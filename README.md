# SGU-MLP: Pytorch Impelementation of the Spatial Gated Multilayer Perceptron
![image](https://zenodo.org/badge/DOI/10.5281/zenodo.15227846.svg)

Pytorch implementation of the SGU-MLP Architecture from the paper "Spatial Gated Multi-Layer Perceptron for Land Use". 
The implementation follows the [implementation of the original authors](https://github.com/aj1365/SGUMLP/blob/main/SGUMLP.ipynb). It differs from the architecture described in the [paper published on GitHub](https://github.com/aj1365/SGUMLP/blob/main/Spatial_Gated_Multi-Layer_Perceptron_for_Land_Use_and_Land_Cover_Mapping.pdf) in the following aspects: 

- Input patches and DWC Block outputs are combined using a residual connection.
- Patches are embedded using a Convolutional Layer (by default, the `embedding_kernel_size` is set to 1 to achieve per pixel projections).
- Input patches can be overlapping (this is only relevant for data preprocessing, not for general model usage).

Additionally, this implementation makes the initial residual weights configurable and learnable. 

An implementation of [MLP Mixer](https://arxiv.org/abs/2105.01601) with optional usage of [Spatial Gated Units](https://arxiv.org/abs/2105.08050) in the MLP Blocks is also included. See `src/sgu_mlp/models.py` for details. 

### Basic Usage

````python
from sgu_mlp import SGUMLPMixer
import torch

height = width = 64
num_patches = height * width
channels = 3
patch_size = 11

input_dimensions = (patch_size, patch_size, channels)
patches = torch.randn(num_patches, *input_dimensions)

# feature extractor
sgu = SGUMLPMixer(
    input_dimensions=input_dimensions,
    token_features=32,
    mixer_features_channel=64,
    mixer_features_sequence=96,
)
out = sgu(patches)

# (num_patches, patch_size**2, channels)
print(out.shape) 

# classifier
num_classes = 8
sgu_clf = SGUMLPMixer(
    input_dimensions=input_dimensions,      
    token_features=32,                      
    mixer_features_channel=64,              
    mixer_features_sequence=96,
    num_classes=num_classes
)
out = sgu_clf(patches)

# (num_patches, num_classes)
print(out.shape)
````

### Installation

If you just want to use SGU-MLP Architecture:
````bash
pip install git+https://github.com/simulacrum6/sgu-mlp.git
````

If you want to run the experiments as well:
````bash
pip install "git+https://github.com/simulacrum6/sgu-mlp.git#egg=sgu-mlp[experiments]"
````

### Running Experiments

To run the replication experiment, first download the benchmark datasets. 
You can download them from [gDrive](https://drive.usercontent.google.com/download?id=1dLJJrNJpQoQeDHybs37iGxmrSU6aP2xv&export=download) or run

````bash
python3 -m experiments.run download --out_dir='/path/to/data/dir'
````

To run the replication experiment:
````bash
python3 -m experiments.run experiment replication
````
Per default, it is assumed, that you run the script from the root of this repository (``--root_dir='data/config')``.

To run a custom experiment:
````bash
python3 -m experiments.run run <experiment_type> <cfg_path>
````
**Arguments**
- `<experiment_type`: "cv" or "ood".
- `<cfg_path>`: path to the config file for the experiment.

See `data/config` for examples. Of the config format.

### References
- _[Spatial Gated Multi-Layer Perceptron for Land Use and Land Cover Mapping](https://doi.org/10.1109/LGRS.2024.3354175)_ - Jamali et al. 2024
- _[Pay Attention to MLPs](https://dl.acm.org/doi/10.5555/3540261.3540965)_ - Liu et al. 2021
- _[MLP-Mixer: An all-MLP Architecture for Vision](https://dl.acm.org/doi/10.5555/3540261.3542118)_ - Tolstikhin et al. 2021

### Citations

When using this software for your research, please cite the orginial article as well as this version of the software.

```bibtex
@article{10399888,
  author={Jamali, Ali and Roy, Swalpa Kumar and Hong, Danfeng and Atkinson, Peter M. and Ghamisi, Pedram},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Spatial-Gated Multilayer Perceptron for Land Use and Land Cover Mapping}, 
  year={2024},
  volume={21},
  number={},
  pages={1-5},
  keywords={Feature extraction;Classification algorithms;Hyperspectral imaging;Data models;Transformers;Biological system modeling;Training data;Attention mechanism;image classification;spatial gating unit (SGU);vision transformers (ViTs)},
  doi={10.1109/LGRS.2024.3354175}}
```

````bibtex
@software{hamacher2024sgumlp,
  title = {SGU-MLP: Pytorch Implementation of the Spatial Gated Multi-Layer Perceptron},
  author = {Hamacher, Marius},
  year = {2025},
  url = {https://github.com/simulacrum6/sgu-mlp},
  version = {0.1.0},
  note = {Pytorch Implementation of the SGU-MLP Architecture from the paper "Spatial Gated Multi-Layer Perceptron for Land Use and Land Cover Mapping"}}
````
