import numpy as np


class Linear_Regression:
    def parameter_init(self, dim):
        self.b = 0
        self.W = np.zeros((dim, 1))

    def predict(self, X):
        return np.dot(X, self.W) + self.b

    def RMSELoss(self, X, Y):
        return np.sqrt(np.mean((Y - self.predict(X)) ** 2))

    def train(self, X, Y, epochs=10000, lr=0.1):
        batch_size = X.shape[0]
        W_dim = X.shape[1]
        self.parameter_init(W_dim)

        lr_b = 0
        lr_W = np.zeros((W_dim, 1))

        for epoch in range(epochs):
            # mse loss
            grad_b = -2 * np.sum(Y - self.predict(X)) / batch_size
            grad_W = -2 * np.dot(X.T, (Y - self.predict(X))) / batch_size
            # adagrad
            lr_b += grad_b ** 2
            lr_W += grad_W ** 2

            # update
            delta_b = lr / np.sqrt(lr_b) * grad_b
            delta_W = lr / np.sqrt(lr_W) * grad_W
            self.b = self.b - delta_b
            self.W = self.W - delta_W
