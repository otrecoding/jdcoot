import numpy as np

from .classif import Classifier


class KRRClassifier(Classifier):

    def __init__(self, lambd=1e-2):
        self.lambd = lambd

    def fit(self, K, y, sw=False):
        ns = K.shape[0]
        if sw:
            K = K * sw
        K0 = np.vstack((np.hstack((np.eye(ns), np.zeros((ns, 1)))), np.zeros((1, ns + 1))))

        ## true reg in RKHS
        # K0=np.vstack((np.hstack((K,np.zeros((ns,1)))),np.zeros((1,ns+1))))

        K1 = np.hstack((K, np.ones((ns, 1))))
        if sw:
            y1 = K1.T.dot(y * sw)
        else:
            y1 = K1.T.dot(y)

        temp = np.linalg.solve(K1.T.dot(K1) + self.lambd * K0, y1)
        self.w, self.b = temp[:-1], temp[-1]

    def predict(self, K):
        return np.dot(K, self.w) + self.b
