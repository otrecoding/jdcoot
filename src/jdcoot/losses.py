import numpy as np


def loss_crossentropy(Y, F):
    """
    Computes the cross-entropy loss between the true labels (Y) and predicted probabilities (F).

    Args:
        Y (numpy.ndarray): A 2D array of shape (n_samples, n_classes) representing the one-hot encoded
                           true labels for each sample.
        F (numpy.ndarray): A 2D array of shape (n_samples, n_classes) representing the predicted
                           probabilities for each class.

    Returns:
        numpy.ndarray: A 2D array of shape (n_samples, n_samples) representing the computed cross-entropy
                       loss for each pair of samples.

    Notes:
        - A small epsilon value (1e-12) is added to the predicted probabilities to avoid numerical
          instability when taking the logarithm.
        - The computation is vectorized using Einstein summation for efficiency.
    """
    eps = 1e-12
    logF = np.log(F + eps)  # Avoid np.array(K.log(...)) if K is not needed
    # Vectorized computation: Y.T @ logF (but adjusted for broadcasting)
    res = -np.einsum("ki,jk->ij", Y, logF)
    return res  # No need for .numpy() if Y and F are already NumPy arrays


def loss_crossentropy2(Y, F):
    """
    Computes the cross-entropy loss between the true labels and predicted probabilities.

    Args:
        Y (numpy.ndarray): A 2D array of true labels, where each row is a one-hot encoded vector.
        F (numpy.ndarray): A 2D array of predicted probabilities, where each row corresponds to
                           the predicted probability distribution for a sample.

    Returns:
        numpy.ndarray: A 1D array containing the cross-entropy loss for each sample.

    Notes:
        - A small epsilon value (1e-12) is added to the predicted probabilities to avoid
          numerical instability when taking the logarithm.
        - The function uses matrix multiplication to compute the loss efficiently.
    """
    eps = 1e-12
    Flog = np.log(F + eps)  # Avoid K.log if not needed
    res = -Y @ Flog.T  # Matrix multiplication (equivalent to K.dot)
    return res


def loss_hinge(Y, F):
    """Computes the squared hinge loss between label and score matrices.

    Args:
        Y: Label matrix of shape (n_samples, n_classes) where each row is a one-hot
            encoded vector of the true labels.
        F: Score matrix of shape (n_models, n_classes) where each row contains
            the model's scores for each class.

    Returns:
        A matrix of shape (n_samples, n_models) where each element represents
        the squared hinge loss for the corresponding sample-model pair.
    """
    # Reshape Y and F for broadcasting: (n_samples, 1, n_classes) and (1, n_models, n_classes)
    hinge = np.maximum(0, 1 - Y[:, np.newaxis, :] * F[np.newaxis, :, :]) ** 2
    return np.sum(hinge, axis=-1)  # Sum over classes (last axis)
