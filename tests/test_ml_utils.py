import numpy as np
import pandas as pd

from models import ml_utils


def test_build_sequences_shape():
    data = np.ones((40, 2))
    labels = np.arange(40)
    seq, target = ml_utils.build_sequences(data, labels, sequence_length=16)
    assert seq.shape[1:] == (16, 2)
    assert len(target) == len(seq)


def test_cross_validate_time_series_smoke():
    X = pd.DataFrame(np.random.randn(30, 3))
    y = pd.Series(np.random.randn(30))

    def builder():
        from sklearn.linear_model import LinearRegression

        return LinearRegression()

    score = ml_utils.cross_validate_time_series(builder, X, y, n_splits=3)
    assert isinstance(score, float)
