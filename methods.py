from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


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
    def __init__(self, n_components, n_mixture):
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


if __name__ == '__main__':
    from toy_data import *
    td = ToyData()
    mm = PCAGaussianMixture(2, 3).fit(td.X)
    X_sim = mm.generate(50, 10)
