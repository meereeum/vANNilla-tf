from __future__ import division

import numpy as np
import pandas as pd

from config import config
from classes.data import DataIO
from classes.model import Model


def doWork(file_in, target_label, outfiles):
    """Resurrect protocols for data preprocessing and artificial neural net model
    construction, and use to generate predictions from input data"""
    # preprocess test data
    df = pd.read_csv(file_in)
    mean = pd.read_csv(outfiles['preprocessing_means'])
    std = pd.read_csv(outfiles['preprocessing_stddevs'])

    data = DataIO(df, target_label, lambda x: DataIO.gaussianNorm(x, mean, std))#,
                  #[-10, 10]) # TODO: code limits ??

    # make predictions
    model = Model(graph_def = outfiles['graph_def'])
    predictions = model.predict(data.stream())

    with open(outfiles['targets'], 'r') as f:
        targets = f.read().strip().split(',')

    prediction_targets = (targets[i] for i in predictions)

    with open(outfiles['predictions'], 'w') as f:
        f.write('\n'.join(prediction_targets) + '\n')

    print 'Predictions saved to: ', outfiles['predictions']

    # if test data is labeled, print accuracy
    if data.n_targets > 0:
        _, y = data.splitXY()
        actual = [np.argmax(one_hot) for one_hot in y.values]
        assert len(actual) == len(predictions)
        correct = sum(y_out == y_actual for y_out, y_actual
                      in zip(predictions, actual))
        print 'accuracy: ', correct/len(predictions)


if __name__ == '__main__':
    doWork(file_in = config.TESTING_DATA,
           target_label = config.TARGET_LABEL,
           outfiles = config.OUTFILES)
