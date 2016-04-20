from __future__ import division

import pandas as pd

from config import config
from data import DataIO, splitTrainValidate
from model import Model
from optimize import GridSearch


#def trainVanillaWithEarlyStopping(): TODO

def trainWithNestedCV(file_in = TRAINING_DATA, target_label = TARGET_LABEL,
                      d_hyperparams = HYPERPARAM_GRID,
                      d_architectures = HIDDEN_LAYER_GRID,
                      seed = None, num_cores = None, verbose = False):
    """Preprocess training data, use grid search of all combinatorial possibilities
    for given hyperparameters and layer architecture to tune artificial neural net
    model, and save files necessary for resurrection of tuned model"""
    KWARGS = {'seed': seed, 'num_cores': num_cores, 'verbose': verbose}

    df = pd.read_csv(file_in)
    assert sum(df.isnull().any()) == False

    # record categorical targets for decoding test set
    targets = list(pd.get_dummies(df[target_label]))
    with open(outfiles['targets'], 'w') as f:
        f.write(','.join(targets))

    # nested validation
    outer_train, outer_validate = splitTrainValidate(df, perc_training=0.8)

    # extract raw features mean, stddev from train set to use for all preprocessing
    raw_features, _ = DataIO(outer_train, target_label).splitXY()
    params = (raw_features.mean(axis=0), raw_features.std(axis=0))
    for k, param in zip(('preprocessing_means', 'preprocessing_stddevs'), params):
         with open(outfiles[k], 'w') as f:
             param.to_csv(f)

    # preprocess features
    train = DataIO(outer_train, target_label, DataIO.gaussianNorm)#, [-10, 10])

    tuner = GridSearch(d_hyperparams, d_architectures)
    hyperparams, architecture = tuner.tuneParams(train, **KWARGS)

    mean, std = params
    validate = DataIO(outer_validate, target_label, lambda x:
                      DataIO.gaussianNorm(x, mean, std))#, [-10, 10])

    hyperparams['epochs'] = 200

    data = {'train': train.stream(batchsize = hyperparams['n_minibatch'],
                                  max_iter = hyperparams['epochs']),
            'validate': validate.stream()}

    model = Model(hyperparams, architecture)

    val_accs = model.train(data, train.len_, logging = True, save = True,
                           outfile = outfiles['graph_def'], seed = seed,
                           num_cores = num_cores, verbose = verbose)

    i, max_ = max(enumerate(val_accs, 1), key = lambda x: (x[1], x[0]))

    print """
Validation accuracies: {}
BEST: {}
(at epoch {})""".format(val_accs, max_, i)

    with open(outfiles['performance'], 'w') as f: # TODO ?
        f.write("Estimated accuracy: {}".format(max_))

    print "Trained model saved to: ", OUTFILES['graph_def']


if __name__ == '__main__':
    trainWithNestedCV(TRAINING_DATA, seed = SEED, num_cores = NUM_CORES,
                      verbose = VERBOSE)
