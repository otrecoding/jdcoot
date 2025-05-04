import math

import numpy as np
import pandas as pd

# -

# # Data Generation

# +

def data_generator(n_Source, n_Target, d_Source, d_Target, mean_X_Source, mean_X_Target, mean_Y_Source,
                   mean_Y_Target, Source_Generation_Correlation, Source_Non_Generation_Correlation,
                   Target_Generation_Correlation, Target_Non_Generation_Correlation,
                   Sparse_Rate,
                   Odds_Ratio_Source, Odds_Ratio_Target, R2_Source, R2_Target,
                   Observed_Covariates_Proportion_Source,
                   Observed_Covariates_Proportion_Target,
                   Indexes_Chosen_For_Generation=None, Cols_Chosen_For_Observation_Source=None,
                   Cols_Chosen_For_Observation_Target=None, Poisson=False):
    """
    data_generator :  Simulate Data from a Scenario with chosen parameters

    Input : n_Source, n_Target (int): number of observation of source/target

            d_Source, d_Target (int): number of variables of source/target (must be the same for the generation in the "same world"/ consider that this set of variable is the "universe")

            mean_X_Source, mean_X_Target (d_Source/d_Target dimensional array of float) : mean of the Normal law of the covariates of source/target

            mean_Y_Source,mean_Y_Target (float) : mean of the continuous objective variable of source/target

            Source_Generation_Correlation, Source_Non_Generation_Correlation (float in  [0,1]): auto correlation coefficient of active/non active covariates for source

            Target_Generation_Correlation, Target_Non_Generation_Correlation (float in  [0,1]): auto correlation coefficient of active/non active covariates for target

            Sparse_Rate (float in  [0,1]) : Proportion of active covariates for generation (Same for Source and Traget because generation in the "same world"/ consider that this generation explains the observed phenomenon)

            Odds_Ratio_Source, Odds_Ratio_Target (float) : Odds ratio of the model of source/target (for probabilities calculation in discrete case)

            R2_Source, R2_Target (float in  [0,1]) : R^2 of the model of source/target (for white noise calculation in continuous case)

            Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target (float in  [0,1]) : Proportion of observed covariates of source/target

            Indexes_Chosen_For_Generation (array of int) : Array of indexes of active variables. (In order to keep the same generation when we want to generate a test sample) If None, chosen randomly

            Cols_Chosen_For_Observation_Source (Array of str :  ['X1', ... , 'Xd']) : Array of names of observed variables in source (In order to keep the same observations when we want to compare performance with test sample) If None, chosen randomly

            Cols_Chosen_For_Observation_Target, :['X1', ... , 'Xd']) : Array of names of observed variables in target (In order to keep the same observations when we want to compare performance with test sample) If None, chosen randomly

            Poisson (Bool) : Uses Poisson modeling to generate more than 2 classes

    Output : Data (tuple): tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                            | X_1 | ... | X_d | Y | Z |

                                        with X_i the ith observed covariate,
                                             Y the continuous objective variable for regression analysis,
                                             Z the discrete objective variable for classification analysis

    """


    # General parameters

    if any(Indexes_Chosen_For_Generation) == None:
        Indexes_Chosen_For_Generation = np.random.choice(np.arange(d_Source), math.ceil(Sparse_Rate * d_Source),
                                                         replace=False)
        # same variables to generate target and source

    # Source covariables generation

    cov_Source = np.eye(d_Source)
    cov_Target = np.eye(d_Target)

    I = 0

    for i in Indexes_Chosen_For_Generation:
        J = 0
        for j in Indexes_Chosen_For_Generation:
            cov_Source[i, j] = Source_Generation_Correlation ** abs(I - J)
            J = J + 1
        I = I + 1

    I = 0

    for i in np.setdiff1d(np.arange(d_Source), Indexes_Chosen_For_Generation):
        J = 0
        for j in np.setdiff1d(np.arange(d_Source), Indexes_Chosen_For_Generation):
            cov_Source[i, j] = Source_Non_Generation_Correlation ** abs(I - J)
            J = J + 1
        I = I + 1

    # cov_Source[Indexes_Chosen_For_Generation,Indexes_Chosen_For_Generation]=Source_Generation_Correlation

    I = 0
    for i in Indexes_Chosen_For_Generation:
        J = 0
        for j in Indexes_Chosen_For_Generation:
            cov_Target[i, j] = Target_Generation_Correlation ** abs(I - J)
            J = J + 1
        I = I + 1

    # MEME NOMBRE DE VARIABLES DANS SOURCE ET DANS TARGET CAR MEME  MONDE
    I = 0

    for i in np.setdiff1d(np.arange(d_Source), Indexes_Chosen_For_Generation):
        J = 0
        for j in np.setdiff1d(np.arange(d_Source), Indexes_Chosen_For_Generation):
            cov_Target[i, j] = Target_Non_Generation_Correlation ** abs(I - J)
            J = J + 1
        I = I + 1

    # print(cov_Source[np.ix_(Indexes_Chosen_For_Generation,Indexes_Chosen_For_Generation)])

    # print(cov_Source[np.ix_(np.setdiff1d(np.arange(0, d_Source - 1),Indexes_Chosen_For_Generation) ,np.setdiff1d(np.arange(0, d_Source - 1),Indexes_Chosen_For_Generation) )])

    X_Source = np.random.multivariate_normal(mean_X_Source, cov_Source, n_Source)

    # Source objective variables generation

    # Continuous
    # Parameters of regression
    # Coefficients for linear predictor

    if sum(mean_X_Source[Indexes_Chosen_For_Generation]) == 0 or mean_Y_Source == 0:
        b_source = 1
    else:
        b_source = mean_Y_Source / sum(mean_X_Source[Indexes_Chosen_For_Generation])

    a_Source = np.zeros(d_Source)
    a_Source[Indexes_Chosen_For_Generation] = b_source

    # Sigma
    sigma_Source = np.var(np.dot(X_Source, a_Source)) * (1 - R2_Source) / R2_Source

    # Variable generation
    Y_Source = np.dot(X_Source, a_Source) + np.random.normal(loc=0, scale=np.sqrt(sigma_Source), size=n_Source)

    # Discrete
    # Parameters of regression
    # Coefficients for linear predictor
    a_Source = np.zeros(d_Source)
    a_Source[Indexes_Chosen_For_Generation] = np.log(Odds_Ratio_Source)

    # Variable generation
    proba_Z_Source = np.exp(np.dot(X_Source, a_Source)) / (1 + np.exp(np.dot(X_Source, a_Source)))

    US = np.random.uniform(0, 1, n_Source)

    Z_Source = np.zeros(n_Source)
    Z_Source[US < proba_Z_Source] = 1

    if Poisson:
        at = np.zeros(d_Source)
        at[Indexes_Chosen_For_Generation] = 10 / 100
        Z_Source = np.random.poisson(np.exp(np.dot(X_Source, at)), n_Source)

    # Target covariables generation
    X_Target = np.random.multivariate_normal(mean_X_Target, cov_Target, n_Target)
    # Target objective variables generation
    # Continuous
    # Parameters of regression
    # Coefficients for linear predictor

    if sum(mean_X_Target[Indexes_Chosen_For_Generation]) == 0 or mean_Y_Target == 0:
        b_Target = 1
    else:
        b_Target = mean_Y_Target / sum(mean_X_Target[Indexes_Chosen_For_Generation])

    a_Target = np.zeros(d_Target)
    a_Target[Indexes_Chosen_For_Generation] = b_Target

    # Sigma
    sigma_Target = np.var(np.dot(X_Target, a_Target)) * (1 - R2_Target) / R2_Target

    # Variable generation
    Y_Target = np.dot(X_Target, a_Target) + np.random.normal(loc=0, scale=np.sqrt(sigma_Target), size=n_Target)

    # Discrete
    # Parameters of regression
    # Coefficients for linear predictor
    a_Target = np.zeros(d_Target)
    a_Target[Indexes_Chosen_For_Generation] = np.log(Odds_Ratio_Target)

    # Variable generation
    proba_Z_Target = np.exp(np.dot(X_Target, a_Target)) / (1 + np.exp(np.dot(X_Target, a_Target)))

    US = np.random.uniform(0, 1, n_Target)

    Z_Target = np.zeros(n_Target)
    Z_Target[US < proba_Z_Target] = 1

    if Poisson:
        at = np.zeros(d_Target)
        at[Indexes_Chosen_For_Generation] = 10 / 100
        Z_Target = np.random.poisson(np.exp(np.dot(X_Target, at)), n_Target)

    if Cols_Chosen_For_Observation_Source is None or Cols_Chosen_For_Observation_Target is None:

        # Source covariables mask (prop% in covariables used for generation + prop% in covariables not used for generation)
        Observed_Covariables_Indexes_Source1 = np.random.choice(Indexes_Chosen_For_Generation,
                                                                math.floor(Observed_Covariates_Proportion_Source * len(
                                                                    Indexes_Chosen_For_Generation)),
                                                                replace=False)

        Observed_Covariables_Indexes_Source2 = np.random.choice(
            np.setdiff1d(np.arange(d_Source), Indexes_Chosen_For_Generation),
            math.floor(Observed_Covariates_Proportion_Source * (d_Source - len(Indexes_Chosen_For_Generation))),
            replace=False)

        X_source_masked = X_Source[:,
                          np.union1d(Observed_Covariables_Indexes_Source1, Observed_Covariables_Indexes_Source2)]

        # Target covariables mask (prop% in covariables used for generation + prop% in covariables not used for generation)
        Observed_Covariables_Indexes_Target1 = np.random.choice(Indexes_Chosen_For_Generation,
                                                                math.floor(Observed_Covariates_Proportion_Target * len(
                                                                    Indexes_Chosen_For_Generation)),
                                                                replace=False)

        Observed_Covariables_Indexes_Target2 = np.random.choice(
            np.setdiff1d(np.arange(d_Target), Indexes_Chosen_For_Generation),
            math.floor(Observed_Covariates_Proportion_Target * (d_Target - len(Indexes_Chosen_For_Generation))),
            replace=False)

        X_target_masked = X_Target[:,
                          np.union1d(Observed_Covariables_Indexes_Target1, Observed_Covariables_Indexes_Target2)]

        # Datasets
        # Source
        data_Source = pd.DataFrame(np.c_[X_source_masked, Y_Source, Z_Source])
        col_names = ['X' + str(i) for i in
                     np.union1d(Observed_Covariables_Indexes_Source1, Observed_Covariables_Indexes_Source2) + 1] + ['Y',
                                                                                                                    'Z']
        data_Source.columns = col_names

        # Target
        data_Target = pd.DataFrame(np.c_[X_target_masked, Y_Target, Z_Target])
        col_names = ['X' + str(i) for i in
                     np.union1d(Observed_Covariables_Indexes_Target1, Observed_Covariables_Indexes_Target2) + 1] + ['Y',
                                                                                                                    'Z']
        data_Target.columns = col_names


    else:
        X_source_masked = X_Source
        X_target_masked = X_Target

        # Datasets
        # Source
        data_Source = pd.DataFrame(np.c_[X_source_masked, Y_Source, Z_Source])
        col_names = ['X' + str(i) for i in np.arange(1, d_Source + 1)] + ['Y', 'Z']
        data_Source.columns = col_names

        # Target
        data_Target = pd.DataFrame(np.c_[X_target_masked, Y_Target, Z_Target])
        col_names = ['X' + str(i) for i in np.arange(1, d_Target + 1)] + ['Y', 'Z']
        data_Target.columns = col_names

        data_Source = data_Source.loc[:, list(Cols_Chosen_For_Observation_Source) + ['Y', 'Z']]
        data_Target = data_Target.loc[:, list(Cols_Chosen_For_Observation_Target) + ['Y', 'Z']]

    return (data_Source, data_Target)
