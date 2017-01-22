from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from RBM import *
import numpy as np
from toy_data import *
import pickle
import copy


class PCAGaussian:
    def __init__(self, n_components):
        self.red = PCA(n_components=n_components)
    def fit(self, X):
        self.n_substance = X.shape[1]
        self.red.fit(X)
        self.X_trans = self.red.transform(X)
        self.C = np.cov(self.X_trans.T)
        self.mu = np.mean(self.X_trans, 0)
        return self
    def generate(self, n_recipes, n_ingredients):
        X_red_gen = \
            np.random.multivariate_normal(self.mu, self.C, size=n_recipes)
        X_gen = np.dot(X_red_gen, self.red.components_)
        X_bin = np.zeros(X_gen.shape)
        for i in range(X_gen.shape[0]):
            ix = np.argsort(X_gen[i, :])[::-1]
            ix = ix[:n_ingredients]
            X_bin[i, ix] = 1
        return X_bin


class PCAGaussianMixture:
    def __init__(self, n_components=2, n_mixture=3):
        self.red = PCA(n_components=n_components)
        self.mixture = GaussianMixture(n_components=n_mixture)
    def fit(self, X):
        self.n_substance = X.shape[1]
        self.red.fit(X)
        self.X_trans = self.red.transform(X)
        self.mixture.fit(self.X_trans)
        return self
    def generate(self, n_recipes, n_ingredients):
        X_red_gen = \
            self.mixture.sample(n_recipes)[0]
        X_gen = np.dot(X_red_gen, self.red.components_)
        X_bin = np.zeros(X_gen.shape)
        for i in range(X_gen.shape[0]):
            ix = np.argsort(X_gen[i, :])[::-1]
            ix = ix[:n_ingredients]
            X_bin[i, ix] = 1
        return X_bin


class DBMgenerator:
    def __init__(self, hlayers=[784, 400, 100] , biases=[-1, -1],
                 niterations=100):
        self.hlayers = hlayers
        self.biases = biases 
        self.niterations = niterations

    def fit(self, X): 
        self.nn = initialize_mnist(X, self.hlayers, self.biases)
        self.nn = self.nn.fit(X, self.niterations)
        return self

    def generate(self, n_recipes, n_ingredients):
        Xgen = np.zeros((n_recipes, self.nn.X[0].shape[1]))
        for i in range(n_recipes):
            print 'generating %d' % i
            for j in range(1, len(self.nn.X)):
                self.nn.gibbs(self.nn.X, j)
            for j in range(len(self.nn.X) - 2, -1, -1):
                self.nn.gibbs(self.nn.X, j)
            Xgen[i] = copy.deepcopy(self.nn.X[0][0, :])
        return Xgen

    def filename(self):
        fn = 'DBMgenerator_' + \
            '_'.join([str(x) for x in self.hlayers]) + \
            '_'.join([str(x) for x in biases]) +\
            '_niterations_%d' % self.niterations
        return fn + '.pkl'

    def save(self, DIR):
        with open(DIR + self.filename(), 'w') as f:
            pickle.dump(self, f)


if __name__ == '__main__':
    fn = '/Users/blythed/work/Data/culinary_classification/'
    fn = fn + 'DBMgenerator_784_400_100-1_-1_niterations_1000.pkl'
    with open(fn) as f:
        dbm_gen = pickle.load(f)
    Xbar = dbm_gen.generate(200, 10)

    if 1:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(Xbar[-1, :].reshape([28, 28]))
        plt.show()
