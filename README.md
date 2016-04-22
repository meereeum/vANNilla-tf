# v[ann]illa-tf

tl;dr --> "vanilla" **a**rtificial **n**eural **n**et classifier, implemented in [TensorFlow](https://github.com/tensorflow/tensorflow)

Modularized for convenient model-building, tuning, and training.

## Usage
(1) Configure model by editing ```config.py```, including hyperparameters (learning rate, dropout, L2-regularization...) and layer architecture, as well as paths/to/data and target label header

<sup> Config may be a simple Config - i.e. single settings for hyperparameters (dictionary) and architecture (list of Layer namedtuples) - OR a GridSearchConfig - i.e. hyperparameter dictionary with lists of multiple settings as values, and architecture dictionary with list of activation functions (```tf.relu```, ```tf.sigmoid```, ```tf.tanh```...) and hidden layer architectures (list of lists of nodes per hidden layer).</sup>

(2) Train and save best model, while logging performance

<sup> Input data must be CSV, with quantitative features and qualitative feature labels.</sup>

```
$ python train.py
```

(3) Test model
```
$ python test.py
```

## Under-the-hood
By default, input features are Gaussian-normalized based on mean & stdev of train set.

Training runs in two modes. If config is flagged as:

* ```Config``` --> ```train.py``` uses early-stopping based on single test/validate split to save optimal model.
* ```GridSearchConfig``` --> ```train.py``` uses nested cross-validation to tune hyperparameters and hidden layer architecture (by selecting optimal settings via k-fold cross-validation), followed by early-stopping with held-out validation set to train and save optimal model.

During training, weights and biases are updated based on cross-entropy cost by the [AdamOptimizer](http://arxiv.org/pdf/1412.6980.pdf) algorithm.

Performance is logged to STDOUT and output file; trained model is saved as a TensorFlow graph_def binary.

```test.py``` restores the trained model, saves predictions to file, and outputs accuracy, if test data is labeled.
