import numpy as np


class ToyData(object):
    def __init__(self, n_recipes=1000, n_substance=100, n_ingredients=10,
                 proportions=False):
        self.n_recipes = n_recipes
        self.n_ingredients = n_ingredients
        self.n_substance = n_substance
        self.generate_recipes()

    def generate_recipes(self):
        X = np.random.rand(self.n_recipes, self.n_substance)
        self.X = np.zeros(X.shape)
        for i in range(X.shape[0]): 
            ix = np.argsort(X[i, :])[:self.n_ingredients]
            self.X[i, ix] = 1


if __name__ == '__main__':
    td = ToyData(1000, 100, 10)
