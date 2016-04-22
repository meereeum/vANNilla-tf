# vANNilla-tf

tl;dr: "Vanilla" **A**rtificial **N**eural **N**etwork classifier, implemented in [TensorFlow](https://github.com/tensorflow/tensorflow)

Modularized for convenient model-building, tuning, and training. WIP!

## Usage
(1) Configure model by editing ```config.py```, including hyperparameters and hidden layer architecture.

<sub> Config may be a simple Config -- i.e. single setting for hyperparameters (dictionary) and architecture (list of Layer namedtuples) -- OR a GridSearchConfig -- i.e. hyperparameter dictionary values with list of multiple settings, and architecture with list of activation functions (```tf.relu```, ```tf.sigmoid```, ```tf.tanh```, ...) and hidden layer architectures (list of lists of nodes per hidden layer)</sub>

(2) Train and save best model, while logging performance.

<sub> Input data must be CSV, with quantitative features and qualitative or quantitative feature labels</sub>

``` $ python train.py```

(3) Test model.
```
$ python test.py
```

## Under-the-hood
Runs in two modes. If config is flagged as:

* ```Config``` --> ```train.py``` uses early-stopping based on single test/validate split to save optimal model.
* ```GridSearchConfig``` --> ```train.py``` uses nested cross-validation to tune hyperparameters and hidden layer architecture (by selecting optimal settings via k-fold cross-validation), followed by early-stopping with held-out validation set to train and save optimal model.

```test.py```
