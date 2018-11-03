import math
import numpy


class GenerativeModel:
    def feature_scaling(self, X, train):
        ret = numpy.zeros(X.shape)
        if train:
            self.m = numpy.zeros(self.dim)
            self.n = numpy.zeros(self.dim)
        else:
            assert self.m.shape[0] == X.shape[1]
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

    def train(self, X, Y):
        assert X.shape[0] == Y.shape[0]

        N = X.shape[0]
        self.dim = X.shape[1]

        self.mean_1 = numpy.zeros((self.dim, 1))
        self.mean_2 = numpy.zeros((self.dim, 1))
        sigma_1 = numpy.zeros((self.dim, self.dim))
        sigma_2 = numpy.zeros((self.dim, self.dim))
        N_1 = 0
        N_2 = 0

        X = self.feature_scaling(X, train=True)

        for idx in range(N):
            f = numpy.reshape(X[idx], (-1, 1))
            if Y[idx] == 0:
                self.mean_1 += f
                N_1 += 1
            elif Y[idx] == 1:
                self.mean_2 += f
                N_2 += 1
            else:
                assert False
        self.mean_1 /= N_1
        self.mean_2 /= N_2

        for idx in range(N):
            f = numpy.reshape(X[idx], (-1, 1))
            if Y[idx] == 0:
                f -= self.mean_1
                s = numpy.dot(f, f.T)
                sigma_1 += s
            elif Y[idx] == 1:
                f -= self.mean_2
                s = numpy.dot(f, f.T)
                sigma_2 += s
            else:
                assert False
        sigma_1 /= N_1
        sigma_2 /= N_2
        assert numpy.array_equal(sigma_1, sigma_1.T)
        assert numpy.array_equal(sigma_2, sigma_2.T)

        sign, self.sigma_1_logdet = numpy.linalg.slogdet(sigma_1)
        assert sign == 1
        self.sigma_1_inv = numpy.linalg.inv(sigma_1)
        sign, self.sigma_2_logdet = numpy.linalg.slogdet(sigma_2)
        assert sign == 1
        self.sigma_2_inv = numpy.linalg.inv(sigma_2)

        self.P_C1 = N_1 / N
        self.P_C2 = N_2 / N

    def save(self, filename):
        numpy.savez(filename,
                    mean_1=self.mean_1, mean_2=self.mean_2,
                    sigma_1_logdet=self.sigma_1_logdet,
                    sigma_1_inv=self.sigma_1_inv,
                    sigma_2_logdet=self.sigma_2_logdet,
                    sigma_2_inv=self.sigma_2_inv,
                    P_C1=self.P_C1, P_C2=self.P_C2,
                    m=self.m, n=self.n)

    @classmethod
    def load(cls, filename):
        model_data = numpy.load(filename)
        ret = cls()
        ret.mean_1 = model_data['mean_1']
        ret.mean_2 = model_data['mean_2']
        ret.sigma_1_logdet = model_data['sigma_1_logdet']
        ret.sigma_1_inv = model_data['sigma_1_inv']
        ret.sigma_2_logdet = model_data['sigma_2_logdet']
        ret.sigma_2_inv = model_data['sigma_2_inv']
        ret.P_C1 = model_data['P_C1']
        ret.P_C2 = model_data['P_C2']
        ret.m = model_data['m']
        ret.n = model_data['n']
        ret.dim = ret.mean_1.shape[0]
        return ret

    def log_f(self, x, mean, sigma_logdet, sigma_inv, D):
        assert x.shape[0] == self.dim
        x_0 = x.copy()
        x_0 -= mean
        z = -0.5 * x_0.T.dot(sigma_inv).dot(x_0)
        log_ret = -D / 2 * math.log(2 * math.pi) + (-0.5) * sigma_logdet + z
        return log_ret

    def inference_one(self, x):
        assert x.shape[0] == self.dim
        P_x_given_C1 = self.log_f(x, self.mean_1, self.sigma_1_logdet, self.sigma_1_inv, self.dim)
        P_x_given_C2 = self.log_f(x, self.mean_2, self.sigma_2_logdet, self.sigma_2_inv, self.dim)
        # Bayes
        P_x_given_C1 -= P_x_given_C2
        try:
            P_x_given_C1 = math.exp(P_x_given_C1)
        except OverflowError:
            return 0
        P_C1_given_x = P_x_given_C1 * self.P_C1 / (P_x_given_C1 * self.P_C1 + 1 * self.P_C2)
        if P_C1_given_x > 0.5:
            return 0
        else:
            return 1

    def inference(self, X):
        X = self.feature_scaling(X, train=False)
        N = X.shape[0]
        ret = numpy.zeros((N, 1))
        for idx in range(N):
            ret[idx] = self.inference_one(numpy.reshape(X[idx], (-1, 1)))
        return ret

    def accuracy(self, X, Y):
        predicted = self.inference(X)
        return numpy.sum(numpy.where(predicted == Y, 1, 0)) / X.shape[0]
