import numpy as np


class FeatureNormalizer(object):
    def __init__(self, feature_matrix=None):
        if feature_matrix is None:
            self.N = 0
            self.mean = 0
            self.S1 = 0
            self.S2 = 0
            self.std = 0
        else:
            self.mean = np.mean(feature_matrix, axis=0)
            self.std = np.std(feature_matrix, axis=0)
            self.N = feature_matrix.shape[0]
            self.S1 = np.sum(feature_matrix, axis=0)
            self.S2 = np.sum(feature_matrix ** 2, axis=0)
            self.finalize()

    def __enter__(self):
        self.N = 0
        self.mean = 0
        self.S1 = 0
        self.S2 = 0
        self.std = 0
        return self

    def __exit__(self, type, value, traceback):
        self.finalize()

    def accumulate(self, stat):
        self.N += stat['N']
        self.mean += stat['mean']
        self.S1 += stat['S1']
        self.S2 += stat['S2']

    def finalize(self):
        # Finalize statistics
        self.mean = self.S1 / self.N
        self.std = np.sqrt((self.N * self.S2 - (self.S1 * self.S1)) / (self.N * (self.N - 1)))

        # In case we have very brain-death material we get std = Nan => 0.0
        self.std = np.nan_to_num(self.std)

        self.mean = np.reshape(self.mean, [1, -1])
        self.std = np.reshape(self.std, [1, -1])

    def normalize(self, feature_matrix):
        return (feature_matrix - self.mean) / self.std
