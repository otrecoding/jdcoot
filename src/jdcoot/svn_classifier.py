import numpy as np
import scipy.optimize as spo

from .classif import Classifier, hinge_squared_reg_bias, hinge_squared_reg


class SVMClassifier(Classifier):

    def __init__(self, lambd=1e-2, bias=False):
        self.lambd = lambd
        self.w = None
        self.bias = bias

        # w = argmin de la loss, f = val min de la loss

    def fit(self, K, y):
        # beware Y is a binary matrix to allow for more general solvers (see JDOT)
        if self.bias:
            K1 = np.hstack((K, np.ones((K.shape[0], 1))))
            self.w = np.zeros((K1.shape[1], y.shape[1]))
            self.w, self.f, self.log = spo.fmin_l_bfgs_b(
                lambda w: hinge_squared_reg_bias(w, X=K1, Y=y, lambd=self.lambd), self.w, maxiter=1000, maxfun=1000)
            self.b = self.w.reshape((K1.shape[1], y.shape[1]))[-1, :]
            self.w = self.w.reshape((K1.shape[1], y.shape[1]))[:-1, :]

        else:
            self.w = np.zeros((K.shape[1], y.shape[1]))  # K.shape[1] = n ou n' , y.shape[1] = nb classes
            self.w, self.f, self.log = spo.fmin_l_bfgs_b(lambda w: hinge_squared_reg(w, X=K, Y=y, lambd=self.lambd),
                                                         self.w, maxiter=1000, maxfun=1000)
            self.w = self.w.reshape((K.shape[1], y.shape[1]))

    def predict(self, K):
        if self.bias:
            return np.dot(K, self.w) + self.b
        else:
            return np.dot(K, self.w)
