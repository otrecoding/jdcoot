import math

import numpy as np
import pandas as pd


class DataScenario:
    r"""
    Sample data from a scenario

    Parameters
    ----------

    - `size_source`, `size_target` : number of observation of source/target
    - `dim_source`, `dim_target` : number of variables of source/target 
    - `mean_x_source`, `mean_x_target` :  mean of the distribution of the co-variates of source/target
    - `mean_y_source`, `mean_y_target` :  mean of the distribution of the continuous objective variable of source/target
    - `generation_correlation_source`, `non_generation_correlation_source` : auto correlation coefficient of active/non active co-variates for source
    - `generation_correlation_target`, `non_generation_correlation_target` : auto correlation coefficient of active/non active co-variates for target
    - `sparse_rate` : proportion of active co-variates for generation (Same for source and target because generation in the "same world"/ consider that this generation explains the observed phenomenon)
    - `odds_ratio_source`, `odds_ratio_target` : odds ratio of the model of source/target (for probabilities calculation in discrete case)
    - `r2_source`, `r2_target` : R^2 of the model of source/target (for white noise calculation in continuous case)
    - `observed_co-variates_proportion_source`, `observed_co-variates_proportion_target` : proportion of observed co-variates of source/target

    """

    def __init__(self):

        self.size_source = 1000
        self.size_target = 1000
        d = 100
        self.dim_source = d
        self.dim_target = d
        pxo = 0.2
        self.observed_covariates_proportion_source = pxo
        self.observed_covariates_proportion_target = pxo
        self.sparse_rate = 0.75

        self.odds_ratio_source = 0.5
        self.odds_ratio_target = 0.5
        self.r2_source = 0.6
        self.r2_target = 0.6
        self.mean_x_source = np.zeros(self.dim_source)
        self.mean_x_target = np.zeros(self.dim_target)
        self.mean_y_source = 0
        self.mean_y_target = 0
        self.generation_correlation_source = 0.7
        self.non_generation_correlation_source = 0.2
        self.generation_correlation_target = 0.7
        self.non_generation_correlation_target = 0.2
        self.poisson = False

    def generate(self, indexes_chosen_for_generation,
                 cols_chosen_for_observation_source = None,
                 cols_chosen_for_observation_target = None):
        """
        - `indexes_chosen_for_generation` : array of indexes of active variables. (In order to keep the same generation when we want to generate a test sample) If None, chosen randomly
        - `cols_chosen_for_observation_source` ['X1', ... , 'Xd'] : names of observed variables in source (In order to keep the same observations when we want to compare performance with test sample) If None, chosen randomly
        - `cols_chosen_for_observation_target` ['X1', ... , 'Xd'] : names of observed variables in target (In order to keep the same observations when we want to compare performance with test sample) If None, chosen randomly
        - `poisson` : Uses Poisson modeling to generate more than 2 classes

        Returns
        -------

        Tuple of 2 dataframes (source, target) with the following format :

        .. math::
                                                | X_1 | ... | X_d | Y | Z |

        - :math:`X_i` the ith observed co-variate,
        - :math:`Y` the continuous objective variable for regression analysis,
        - :math:`Z` the discrete objective variable for classification analysis

        """

        # source co-variables generation

        cov_source = np.eye(self.dim_source)
        cov_target = np.eye(self.dim_target)

        k = 0
        for i in indexes_chosen_for_generation:
            l = 0
            for j in indexes_chosen_for_generation:
                cov_source[i, j] = self.generation_correlation_source ** abs(k - l)
                l += 1
            k += 1

        k = 0

        for i in np.setdiff1d(np.arange(self.dim_source), indexes_chosen_for_generation):
            l = 0
            for j in np.setdiff1d(np.arange(self.dim_source), indexes_chosen_for_generation):
                cov_source[i, j] = self.non_generation_correlation_source ** abs(k - l)
                l += 1
            k += 1

        k = 0
        for i in indexes_chosen_for_generation:
            l = 0
            for j in indexes_chosen_for_generation:
                cov_target[i, j] = self.generation_correlation_target ** abs(k - l)
                l += 1
            k += 1

        # MEME NOMBRE DE VARIABLES DANS SOURCE ET DANS TARGET CAR MEME  MONDE
        k = 0
        for i in np.setdiff1d(np.arange(self.dim_source), indexes_chosen_for_generation):
            l = 0
            for j in np.setdiff1d(np.arange(self.dim_source), indexes_chosen_for_generation):
                cov_target[i, j] = self.non_generation_correlation_target ** abs(k - l)
                l += 1
            k += 1

        x_source = np.random.multivariate_normal(self.mean_x_source, cov_source, self.size_source)

        # source objective variables generation

        # Continuous
        # Parameters of regression
        # Coefficients for linear predictor

        if sum(self.mean_x_source[indexes_chosen_for_generation]) == 0 or self.mean_x_source == 0:
            b_source = 1
        else:
            b_source = self.mean_y_source / sum(self.mean_x_source[indexes_chosen_for_generation])

        a_source = np.zeros(self.dim_source)
        a_source[indexes_chosen_for_generation] = b_source

        # Sigma
        sigma_source = np.var(np.dot(x_source, a_source)) * (1 - self.r2_source) / self.r2_source

        # Variable generation
        y_source = np.dot(x_source, a_source) + np.random.normal(loc=0, scale=np.sqrt(sigma_source),
                                                                 size=self.size_source)

        # Discrete
        # Parameters of regression
        # Coefficients for linear predictor
        a_source = np.zeros(self.dim_source)
        a_source[indexes_chosen_for_generation] = np.log(self.odds_ratio_source)

        # Variable generation
        proba_z_source = np.exp(np.dot(x_source, a_source)) / (1 + np.exp(np.dot(x_source, a_source)))

        us = np.random.uniform(0, 1, self.size_source)

        z_source = np.zeros(self.size_source)
        z_source[us < proba_z_source] = 1

        if self.poisson:
            at = np.zeros(self.dim_source)
            at[indexes_chosen_for_generation] = 10 / 100
            z_source = np.random.poisson(np.exp(np.dot(x_source, at)), self.size_source)

        # target co-variables generation
        x_target = np.random.multivariate_normal(self.mean_x_target, cov_target, self.size_target)
        # target objective variables generation
        # Continuous
        # Parameters of regression
        # Coefficients for linear predictor

        if sum(self.mean_x_target[indexes_chosen_for_generation]) == 0 or self.mean_y_target == 0:
            b_target = 1
        else:
            b_target = self.mean_y_target / sum(self.mean_x_target[indexes_chosen_for_generation])

        a_target = np.zeros(self.dim_target)
        a_target[indexes_chosen_for_generation] = b_target

        # Sigma
        sigma_target = np.var(np.dot(x_target, a_target)) * (1 - self.r2_target) / self.r2_target

        # Variable generation
        y_target = np.dot(x_target, a_target) + np.random.normal(loc=0, scale=np.sqrt(sigma_target), size=self.size_target)

        # Discrete
        # Parameters of regression
        # Coefficients for linear predictor
        a_target = np.zeros(self.dim_target)
        a_target[indexes_chosen_for_generation] = np.log(self.odds_ratio_target)

        # Variable generation
        proba_z_target = np.exp(np.dot(x_target, a_target)) / (1 + np.exp(np.dot(x_target, a_target)))

        us = np.random.uniform(0, 1, self.size_target)

        z_target = np.zeros(self.size_target)
        z_target[us < proba_z_target] = 1

        if self.poisson:
            at = np.zeros(self.dim_target)
            at[indexes_chosen_for_generation] = 10 / 100
            z_target = np.random.poisson(np.exp(np.dot(x_target, at)), self.size_target)

        if cols_chosen_for_observation_source is None or cols_chosen_for_observation_target is None:

            # source co-variables mask (prop% in co-variables used for generation + prop% in co-variables not used for generation)
            observed_covariables_indexes_source1 = np.random.choice(indexes_chosen_for_generation,
                                                                    math.floor(
                                                                        self.observed_covariates_proportion_source * len(
                                                                            indexes_chosen_for_generation)),
                                                                    replace=False)

            observed_covariables_indexes_source2 = np.random.choice(
                np.setdiff1d(np.arange(self.dim_source), indexes_chosen_for_generation),
                math.floor(
                    self.observed_covariates_proportion_source * (
                            self.dim_source - len(indexes_chosen_for_generation))),
                replace=False)

            x_source_masked = x_source[:,
                              np.union1d(observed_covariables_indexes_source1, observed_covariables_indexes_source2)]

            # target co-variables mask (prop% in co-variables used for generation + prop% in co-variables not used for generation)
            observed_covariables_indexes_target1 = np.random.choice(indexes_chosen_for_generation,
                                                                    math.floor(
                                                                        self.observed_covariates_proportion_target * len(
                                                                            indexes_chosen_for_generation)),
                                                                    replace=False)

            observed_covariables_indexes_target2 = np.random.choice(
                np.setdiff1d(np.arange(self.dim_target), indexes_chosen_for_generation),
                math.floor(self.observed_covariates_proportion_target * (self.dim_target - len(indexes_chosen_for_generation))),
                replace=False)

            x_target_masked = x_target[:,
                              np.union1d(observed_covariables_indexes_target1, observed_covariables_indexes_target2)]

            # Datasets
            # source
            data_source = pd.DataFrame(np.c_[x_source_masked, y_source, z_source])
            col_names = ['X' + str(i) for i in
                         np.union1d(observed_covariables_indexes_source1, observed_covariables_indexes_source2) + 1] + [
                            'Y',
                            'Z']
            data_source.columns = col_names

            # target
            data_target = pd.DataFrame(np.c_[x_target_masked, y_target, z_target], columns= ['X' + str(i) for i in
                         np.union1d(observed_covariables_indexes_target1, observed_covariables_indexes_target2) + 1] + [
                            'Y',
                            'Z'])


        else:
            x_source_masked = x_source
            x_target_masked = x_target

            # Datasets
            # source
            data_source = pd.DataFrame(np.c_[x_source_masked, y_source, z_source], columns =
             ['X' + str(i) for i in np.arange(1, self.dim_source + 1)] + ['Y', 'Z'])

            # target
            data_target = pd.DataFrame(np.c_[x_target_masked, y_target, z_target],
                                       columns = ['X' + str(i) for i in np.arange(1, self.dim_target + 1)] + ['Y', 'Z'])

            data_source = data_source.loc[:, list(cols_chosen_for_observation_source) + ['Y', 'Z']]
            data_target = data_target.loc[:, list(cols_chosen_for_observation_target) + ['Y', 'Z']]

        return data_source, data_target


class DataScenarioTest(DataScenario):

    """
    Sample a test sample for the reference scenario (300 observations)
    """

    def __init__(self):

        super().__init__()
        self.size_source = 300
        self.size_target = 300
        pxo = 1
        self.observed_covariates_proportion_source = pxo
        self.observed_covariates_proportion_target = pxo



class DataScenarioPoisson(DataScenario):

    """
    Sample a reference scenario but with more than 2 classes for the classification analysis
    """

    def __init__(self):
        super().__init__()
        self.size_source = 10000
        self.size_target = 10000
        self.poisson = True


class DataScenarioPoissonTest(DataScenario):

    """
    Sample a test sample for the Poisson reference scenario (3000 observations)
    """

    def __init__(self):
        super().__init__()
        self.size_source = 3000
        self.size_target = 3000
        self.poisson = True
        pxo = 1
        self.observed_covariates_proportion_source = pxo
        self.observed_covariates_proportion_target = pxo
