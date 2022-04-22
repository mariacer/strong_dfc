# Deep Feedback Control

Implementation of Deep Feedback Control and some extensions. The basis of this repository is a cleaned-up version of this 
[public repository](https://github.com/meulemansalex/deep_feedback_control/tree/main).

## Requirements

The necessary conda environment to run the code is provided in the file [dfc_environment.yml](dfc_environment.yml). To generate the environment type `conda env create -f dfc_environment.yml` and activate it with `conda activate dfc_environment`.

Note that you'll also need to install the [hypnettorch library](https://github.com/chrhenning/hypnettorch) by doing `python3 -m pip install hypnettorch`.

## Documentation

How to open the documentation is explained in the `docs` folder.

## Running experiments

For running experiments, move to the `dfc` subfolder. Further instructions can be found on the [README](dfc/README.rst) there. Command line arguments with good hyperparameter settings for the different DFC variants can be found in the [EXPERIMENTS.rst](dfc/EXPERIMENTS.rst) file.

## Citation

When using this package in your research project, please consider citing one of our papers for which this package has been developed.

```
@inproceedings{Meulemans2021Dec,
   title={Credit Assignment in Neural Networks through Deep Feedback Control},
   author={Alexander Meulemans and Matilde Tristany Farinha and Javier Garcia Ordonez and Pau Vilimelis Aceituno and Joao Sacramento and Benjamin F. Grewe},
   booktitle={Advances in Neural Information Processing Systems},
   year={2021},
   url={https://arxiv.org/abs/2106.07887}
}
```

```
@misc{https://doi.org/10.48550/arxiv.2204.07249,
  title = {Minimizing Control for Credit Assignment with Strong Feedback},
  author = {Meulemans, Alexander and Farinha, Matilde Tristany and Cervera, Maria R. and Sacramento, João and Grewe, Benjamin F.},
  publisher = {arXiv},
  year = {2022},
  url = {https://arxiv.org/abs/2204.07249},
}

```
