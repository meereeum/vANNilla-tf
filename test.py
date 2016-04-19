import numpy as np
import pandas as pd

from data import DataIO
from config import OUTFILES, TARGET_LABEL, TESTING_DATA
from model import Model


def doWork(file_in):
    """Resurrect protocols for data preprocessing and artificial neural net model
    construction, and use to generate predictions from input data"""
    df = pd.read_csv(file_in)
    mean = pd.read_csv(OUTFILES['preprocessing_means'])
    std = pd.read_csv(OUTFILES['preprocessing_stddevs'])

    data = DataIO(df, TARGET_LABEL, lambda x: DataIO.gaussianNorm(x, mean, std))#,
                  #[-10, 10]) # TODO: code limits ??

    model = Model(graph_def = OUTFILES['graph_def'])
    predictions = model.predict(data.stream())

    with open(OUTFILES['targets'], 'r') as f:
        targets = f.read().strip().split(',')

    prediction_targets = (targets[i] for i in predictions)

    with open(OUTFILES['predictions'], 'w') as f:
        f.write('\n'.join(prediction_targets) + '\n')

    print "Predictions saved to: {}".format(OUTFILES['predictions'])

    # if test data is labeled, print accuracy
    if data.n_targets > 0:
        _, y = data.splitXY()
        actual = [np.argmax(one_hot) for one_hot in y.values]
        assert len(actual) == len(predictions)
        correct = sum(y_out == y_actual for y_out, y_actual
                      in zip(predictions, actual))
        print "accuracy: ", correct/len(predictions)

if __name__ == '__main__':
    doWork(TESTING_DATA)
