# Stochastic Blockmodels meet Graph Neural Networks

This is a PyTorch implementation of the paper [Stochastic Blockmodels meet Graph Neural Networks](https://proceedings.mlr.press/v97/mehta19a.html).

This repository originates from the authors' implementation of the paper in TensorFlow, which can be found [here](https://github.com/nikhil-dce/SBM-meet-GNN).

## Requirements

The dependencies for this project are listed in `requirements.txt`. To install them, run:

```
pip install -r requirements.txt
```

## Datasets

The datasets currently supported in this project are:

- Cora
- Citeseer

which can be located in the `data` directory.

## Running the code

To train and test the model, you may find the following script useful:

```
bash script/train.sh
```

All available options and hyperparameters can be found in the [args.py](args.py) file, which mostly comply with the authors' original implementation.

## Visualizing the results

After training the model, you can visualize the learned latent communities as well as the node members within using the [notebook](visualize.ipynb).
