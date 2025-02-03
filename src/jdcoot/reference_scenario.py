import numpy as np

from .data_generation import data_generator
from .globals import INDEX_GENERATION

# -

# ## Reference Scenario

# +


# @deprecated
def Sref(Indexes_Chosen_For_Generation=INDEX_GENERATION):
    """
    Sref : Sample a reference scenario as described in the paper

    Input : Indexes_Chosen_For_Generation (array of int) : Array of indexes of active variables. (In order to keep the same generation when we want to generate a test sample) If None, chosen randomly

    Output : Data (tuple): tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

    .. math::
                                            | X_1 | ... | X_d | Y | Z |

    - :math:`X_i` the ith observed covariate,
    - `Y` the continuous objective variable for regression analysis,
    - `Z` the discrete objective variable for classification analysis
    """

    n_S = 1000
    n_T = 1000
    d = 100
    d_S = d
    d_T = d
    pxo = 0.2
    pxo_S = pxo
    pxo_T = pxo
    sr = 0.75

    OR_S = 0.5
    OR_T = 0.5
    R2_S = 0.6
    R2_T = 0.6
    mX_S = np.zeros(d_S)
    mX_T = np.zeros(d_T)
    mY_S = 0
    mY_T = 0
    # cov_S = np.eye(d_S)
    # cov_T = np.eye(d_T)
    rho_source_generation = 0.7
    rho_source_non_generation = 0.2
    rho_target_generation = 0.7
    rho_target_non_generation = 0.2

    return (data_generator(n_S, n_T, d_S, d_T, mX_S, mX_T, mY_S, mY_T, rho_source_generation, rho_source_non_generation,
                           rho_target_generation, rho_target_non_generation, sr, OR_S, OR_T, R2_S, R2_T,
                           pxo_S, pxo_T, Indexes_Chosen_For_Generation))



# @deprecated
def Sref_test(Indexes_Chosen_For_Generation=INDEX_GENERATION):
    """
    Sref_test : Sample a test sample for the reference scenario (300 observations)

    Input : Indexes_Chosen_For_Generation (array of int) : Array of indexes of active variables. (In order to keep the same generation when we want to generate a test sample) If None, chosen randomly

    Output : Data (tuple): tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

    .. math::

                                                | X_1 | ... | X_d | Y | Z |

    - :math:`X_i` the ith observed covariate,
    - :math:`Y` the continuous objective variable for regression analysis,
    - :math:`Z` the discrete objective variable for classification analysis
    """

    n_S = 300
    n_T = 300
    d = 100
    d_S = d
    d_T = d
    pxo = 1
    sr = 0.75
    pxo_S = pxo
    pxo_T = pxo
    OR_S = 0.5
    OR_T = 0.5
    R2_S = 0.6
    R2_T = 0.6
    mX_S = np.zeros(d_S)
    mX_T = np.zeros(d_T)
    mY_S = 0
    mY_T = 0
    # cov_S = np.eye(d_S)
    # cov_T = np.eye(d_T)
    rho_source_generation = 0.7
    rho_source_non_generation = 0.2
    rho_target_generation = 0.7
    rho_target_non_generation = 0.2

    return (data_generator(n_S, n_T, d_S, d_T, mX_S, mX_T, mY_S, mY_T, rho_source_generation, rho_source_non_generation,
                           rho_target_generation, rho_target_non_generation, sr, OR_S, OR_T, R2_S, R2_T,
                           pxo_S, pxo_T, Indexes_Chosen_For_Generation))


# -

# ## Reference scenario in case of multiclassification

# +



# @deprecated
def Sref_Poisson(Indexes_Chosen_For_Generation=INDEX_GENERATION):
    """
    Sref_Poisson : Sample a reference scenario but with more than 2 classes for the classification analysis

    Input : Indexes_Chosen_For_Generation (array of int) : Array of indexes of active variables. (In order to keep the same generation when we want to generate a test sample) If None, chosen randomly

    Output : Data (tuple): tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

    .. math::

                                            | X_1 | ... | X_d | Y | Z |

    - :math:`X_i` the ith observed covariate,
    - :math:`Y` the continuous objective variable for regression analysis,
    - :math:`Z` the discrete objective variable for classification analysis
    """

    n_S = 10000
    n_T = 10000
    d = 100
    d_S = d
    d_T = d
    pxo = 0.2
    pxo_S = pxo
    pxo_T = pxo
    sr = 0.75
    OR_S = 0.5
    OR_T = 0.5
    R2_S = 0.6
    R2_T = 0.6
    mX_S = np.zeros(d_S)
    mX_T = np.zeros(d_T)
    mY_S = 0
    mY_T = 0
    # cov_S = np.eye(d_S)
    # cov_T = np.eye(d_T)
    rho_source_generation = 0.7
    rho_source_non_generation = 0.2
    rho_target_generation = 0.7
    rho_target_non_generation = 0.2

    return (data_generator(n_S, n_T, d_S, d_T, mX_S, mX_T, mY_S, mY_T, rho_source_generation, rho_source_non_generation,
                           rho_target_generation, rho_target_non_generation, sr, OR_S, OR_T, R2_S, R2_T,
                           pxo_S, pxo_T, Indexes_Chosen_For_Generation, Poisson=True))



# @deprecated
def Sref_Poisson_test(Indexes_Chosen_For_Generation=INDEX_GENERATION):
    """
    Sref_Poisson_test : Sample a test sample for the Poisson reference scenario (3000 observations)

    Input : Indexes_Chosen_For_Generation (array of int) : Array of indexes of active variables. (In order to keep the same generation when we want to generate a test sample) If None, chosen randomly

    Output : Data (tuple): tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

    .. math:

                                            | X_1 | ... | X_d | Y | Z |

    - :math:`X_i` the ith observed covariate,
    - :math:`Y` the continuous objective variable for regression analysis,
    - :math:`Z` the discrete objective variable for classification analysis
    """

    n_S = 3000
    n_T = 3000
    d = 100
    d_S = d
    d_T = d
    pxo = 1
    pxo_S = pxo
    pxo_T = pxo
    sr = 0.75
    OR_S = 0.5
    OR_T = 0.5
    R2_S = 0.6
    R2_T = 0.6
    mX_S = np.zeros(d_S)
    mX_T = np.zeros(d_T)
    mY_S = 0
    mY_T = 0
    # cov_S = np.eye(d_S)
    # cov_T = np.eye(d_T)
    rho_source_generation = 0.7
    rho_source_non_generation = 0.2
    rho_target_generation = 0.7
    rho_target_non_generation = 0.2

    return (data_generator(n_S, n_T, d_S, d_T, mX_S, mX_T, mY_S, mY_T, rho_source_generation, rho_source_non_generation,
                           rho_target_generation, rho_target_non_generation, sr, OR_S, OR_T, R2_S, R2_T,
                           pxo_S, pxo_T, Indexes_Chosen_For_Generation, Poisson=True))
