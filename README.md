# vANNilla-tf

tl;dr: "Vanilla" **A**rtificial **N**eural **N**etwork classifier, implemented in TensorFlow

Modularized for convenient model-building, tuning, and training. WIP!

## Usage
(1) Set configuration in ```config.py```. Input data must be CSV, with quantitative features and qualitative or quantitative feature labels.

## Under-the-hood
Runs in two modes. If config is flagged as:

* ```Config``` --> ```train.py``` uses early-stopping based on single test/validate split to save optimal model.
* ```GridSearchConfig``` --> ```train.py``` uses nested cross-validation to tune hyperparameters and hidden layer architecture (by selecting optimal settings via k-fold cross-validation), followed by early-stopping with held-out validation data to train and save optimal model.

```test.py```
