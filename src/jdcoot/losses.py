import numpy as np
from tf_keras import backend as K


def loss_crossentropy(Y, F):
    eps = 1e-12
    res = np.zeros((Y.shape[0], F.shape[0]))
    logF = np.array(K.log(F + eps))
    for i in range(Y.shape[1]):
        res += -Y[:, i].reshape((Y.shape[0], 1)) * logF[:, i].reshape((1, F.shape[0]))

    return res.numpy()


def loss_crossentropy2(Y, F):
    eps = 1e-12
    Flog = K.log(F + eps)
    # loss calculation based on double sum (sum_ij (ys^i, ypred_t^j))
    res = -K.dot(K.variable(Y), K.transpose(Flog))

    return res.numpy()


def loss_hinge(Y, F):
    res = np.zeros((Y.shape[0], F.shape[0]))
    for i in range(Y.shape[1]):
        res += (
                np.maximum(
                    0,
                    1 - Y[:, i].reshape((Y.shape[0], 1)) * F[:, i].reshape((1, F.shape[0])),
                )
                ** 2
        )
    return res.numpy()
