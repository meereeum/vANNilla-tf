from __future__ import division

import pandas as pd

from config import config
from classes.data import DataIO, splitTrainValidate
from classes.model import Model
from classes.tune import GridSearch


def preprocess(file_in, target_label, outfiles):
    """Generate processed training and validation data from input file, and
    save files necessary to process future test data.

    Returns: (train, validate) as preprocessed DataIO objects
    """
    df = pd.read_csv(file_in)
    assert sum(df.isnull().any()) == False

    # record categorical targets for decoding test set
    targets = list(pd.get_dummies(df[target_label]))
    with open(outfiles['targets'], 'w') as f:
        f.write(','.join(targets))

    # nested validation
    train, validate = splitTrainValidate(df, perc_training=0.8)

    # extract raw features mean, stddev from train set to use for all preprocessing
    raw_features, _ = DataIO(train, target_label).splitXY()
    params = (raw_features.mean(axis=0), raw_features.std(axis=0))
    for k, param in zip(('preprocessing_means', 'preprocessing_stddevs'), params):
         with open(outfiles[k], 'w') as f:
             param.to_csv(f)

    # preprocess features
    return (DataIO(dataset, target_label, lambda x:
                   DataIO.gaussianNorm(x, *params))#, [-10, 10]
            for dataset in (train, validate))

def trainWithEarlyStopping(train, validate, hyperparams, architecture, outfiles,
                           seed = None, num_cores = None, verbose = False):
    """Build trained artificial neural net model using early-stopping with
    validation set, and save files necessary for resurrection of tuned model
    """
    # if not already set, size input & output nodes by data shape
    if not architecture[0].nodes:
        architecture[0] = architecture[0]._replace(nodes = train.n_features)
    if not architecture[-1].nodes:
        architecture[-1] = architecture[-1]._replace(nodes = train.n_targets)

    model = Model(hyperparams, architecture)

    data = {'train': train.stream(batchsize = hyperparams['n_minibatch'],
                                  max_iter = hyperparams['epochs']),
            'validate': validate.stream()}

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

    print "Trained model saved to: ", outfiles['graph_def']

def trainWithNestedCV(train, validate, d_hyperparams, d_architectures,
                      outfiles, seed = None, num_cores = None, verbose = False):
    """Implement nested cross-validation to (1) use grid search of all
    combinatorial possibilities for given hyperparameters and layer architecture
    to tune artificial neural net model, and (2) generate trained model using
    early-stopping with held-out validation set
    """
    KWARGS = {'seed': seed, 'num_cores': num_cores, 'verbose': verbose}

    # choose optimal hyperparams, architecture with k-fold cross-validation
    tuner = GridSearch(d_hyperparams, d_architectures)
    hyperparams, architecture = tuner.tuneParams(train, **KWARGS)

    # train on full training set and use validation set for early-stopping
    hyperparams['epochs'] += 100
    trainWithEarlyStopping(train, validate, hyperparams, architecture, outfiles,
                           **KWARGS)


if __name__ == '__main__':
    train, validate = preprocess(file_in = config.TRAINING_DATA,
                                 target_label = config.TARGET_LABEL,
                                 outfiles = config.OUTFILES)
    try:
        trainWithNestedCV(train = train,
                          validate = validate,
                          d_hyperparams = config.HYPERPARAM_GRID,
                          d_architectures = config.HIDDEN_LAYER_GRID,
                          outfiles = config.OUTFILES,
                          seed = config.SEED,
                          num_cores = config.NUM_CORES,
                          verbose = config.VERBOSE)
    except(AttributeError):
        try:
            trainWithEarlyStopping(train = train,
                                   validate = validate,
                                   hyperparams = config.HYPERPARAMS,
                                   architecture = config.LAYERS,
                                   outfiles = config.OUTFILES,
                                   seed = config.SEED,
                                   num_cores = config.NUM_CORES,
                                   verbose = config.VERBOSE)
        except(AttributeError):
            raise('Must supply valid config for Model training')
