import numpy


class Logistic_Regression:
    def parameter_init(self, dim):
        self.b = 0
        self.W = numpy.zeros((dim, 1))

    def predict(self, X):
        z = numpy.dot(X, self.W) + self.b
        sigma_z = 1 / (1 + numpy.exp(-z))
        return sigma_z

    def CrossEntropyLoss(self, X, Y):
        predicted = self.predict(X)
        return -numpy.sum(Y * numpy.log(predicted) + (1 - Y) * numpy.log(1 - predicted))

    def accuracy(self, X, Y):
        predicted = numpy.where(self.predict(X) > 0.5, 1, 0)
        return numpy.sum(numpy.where(predicted == Y, 1, 0)) / X.shape[0]

    def feature_scaling(self, X, train):
        dim = self.W.shape[0]
        assert X.shape[1] == dim
        ret = numpy.zeros(X.shape)
        if train:
            self.m = numpy.zeros(dim)
            self.n = numpy.zeros(dim)
        for idx in range(X.shape[1]):
            if train:
                m = numpy.min(X[..., idx])
                n = numpy.max(X[..., idx])
                self.m[idx] = m
                self.n[idx] = n
            else:
                m = self.m[idx]
                n = self.n[idx]
            ret[..., idx] = (X[..., idx] - m) / (n - m)
        return ret

    def train(self, X, Y, epochs=10000, lr=0.5):
        batch_size = X.shape[0]
        W_dim = X.shape[1]
        self.parameter_init(W_dim)

        X = self.feature_scaling(X, train=True)

        lr_b = 0
        lr_W = numpy.zeros((W_dim, 1))

        for epoch in range(epochs):
            # cross entropy loss
            predicted = self.predict(X)
            grad_b = -numpy.sum(Y - predicted) / batch_size
            grad_W = -numpy.dot(X.T, (Y - predicted)) / batch_size
            # adagrad
            lr_b += grad_b ** 2
            lr_W += grad_W ** 2

            # update
            delta_b = lr / numpy.sqrt(lr_b) * grad_b
            delta_W = lr / numpy.sqrt(lr_W) * grad_W
            self.b = self.b - delta_b
            self.W = self.W - delta_W

    def save(self, filename):
        numpy.savez(filename, b=self.b, w=self.W, m=self.m, n=self.n)

    @classmethod
    def load(cls, filename):
        model_data = numpy.load(filename)
        model = cls()
        model.b = model_data['b']
        model.W = model_data['w']
        model.m = model_data['m']
        model.n = model_data['n']
        return model
