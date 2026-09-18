import math

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm


def sample(x, px):
    return np.random.choice(x, math.floor(px * len(x)), replace=False)


class DataScenario:
    r"""
    Sample data from a scenario

    Parameters
    ----------

    - `size_source`, `size_target` : number of obs of source/target
    - `dim_source`, `dim_target` : number of variables of source/target
    - `mean_x_source`, `mean_x_target` :  mean of the distribution of the co-variates of source/target
    - `mean_y_source`, `mean_y_target` :  mean of the distribution of the continuous objective variable of source/target
    - `active_autocorr_source`, `inactive_autocorr_source` : auto correlation coefficient of active/non active co-variates for source
    - `active_autocorr_target`, `inactive_autocorr_target` : auto correlation coefficient of active/non active co-variates for target
    - `sparse_rate` : proportion of active co-variates for generation (Same for source and target because generation in the "same world"/ consider that this generation explains the observed phenomenon)
    - `odds_ratio_source`, `odds_ratio_target` : odds ratio of the model of source/target (for probabilities calculation in discrete case)
    - `r2_source`, `r2_target` : R^2 of the model of source/target (for white noise calculation in continuous case)
    - `obs_covar_prop_source`, `obs_covar_prop_target` : proportion of observed co-variates of source/target

    """

    def __init__(self):
        self.size_source = 1000
        self.size_target = 1000
        self.dim_source = 100
        self.dim_target = 100
        self.obs_covar_prop_source = 0.4
        self.obs_covar_prop_target = 0.4

        self.odds_ratio_source = 0.5
        self.odds_ratio_target = 0.5
        self.r2_source = 0.6
        self.r2_target = 0.6
        self.mean_x_source = np.zeros(self.dim_source)
        self.mean_x_target = np.zeros(self.dim_target)
        self.mean_y_source = 0
        self.mean_y_target = 0
        self.active_autocorr_source = 0.3
        self.inactive_autocorr_source = 0.3
        self.active_autocorr_target = 0.3
        self.inactive_autocorr_target = 0.3

    def generate(self, indices):
        """
        - `indices` : array of indices of active variables. (In order to keep the same generation when we want to generate a test sample)

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

        actives = indices
        inactives = np.setdiff1d(np.arange(self.dim_source), actives)

        for kk, ii in enumerate(actives):
            for ll, jj in enumerate(actives):
                cov_source[ii, jj] = self.active_autocorr_source ** abs(kk - ll)

        for kk, ii in enumerate(inactives):
            for ll, jj in enumerate(inactives):
                cov_source[ii, jj] = self.inactive_autocorr_source ** abs(kk - ll)

        for kk, ii in enumerate(actives):
            for ll, jj in enumerate(actives):
                cov_target[ii, jj] = self.active_autocorr_target ** abs(kk - ll)

        for kk, ii in enumerate(inactives):
            for ll, jj in enumerate(inactives):
                cov_target[ii, jj] = self.inactive_autocorr_target ** abs(kk - ll)

        x_source = np.random.multivariate_normal(
            self.mean_x_source, cov_source, self.size_source
        )
        x_target = np.random.multivariate_normal(
            self.mean_x_target, cov_target, self.size_target
        )

        data_source = pd.DataFrame(
            x_source, columns=[f"X{i+1}" for i in range(self.dim_source)]
        )
        data_target = pd.DataFrame(
            x_target, columns=[f"X{i+1}" for i in range(self.dim_target)]
        )

        obs_covariables_ix_source1 = sample(actives, self.obs_covar_prop_source)
        obs_covariables_ix_source2 = sample(inactives, self.obs_covar_prop_source)
        obs_covariables_ix_source = np.union1d(
            obs_covariables_ix_source1, obs_covariables_ix_source2
        )

        obs_covariables_ix_target1 = sample(actives, self.obs_covar_prop_target)
        obs_covariables_ix_target2 = sample(inactives, self.obs_covar_prop_target)
        obs_covariables_ix_target = np.union1d(
            obs_covariables_ix_target1, obs_covariables_ix_target2
        )

        selected_obs_source = [f"X{i+1}" for i in obs_covariables_ix_source]
        selected_obs_target = [f"X{i+1}" for i in obs_covariables_ix_target]

        data_source = data_source.loc[:, list(selected_obs_source)]
        data_target = data_target.loc[:, list(selected_obs_target)]

        b_source = 1

        a_source = np.zeros(self.dim_source)
        a_source[actives] = b_source

        y_source = np.dot(x_source, a_source)

        sigma_source = np.var(y_source) * (1 - self.r2_source) / self.r2_source

        y_source += np.random.normal(
            loc=0, scale=np.sqrt(sigma_source), size=self.size_source
        )

        a_source = np.zeros(self.dim_source)
        a_source[actives] = np.log(self.odds_ratio_source)

        z = np.dot(x_source, a_source)
        proba_z_source = np.exp(z) / (1 + np.exp(z))

        us = np.random.uniform(0, 1, self.size_source)

        z_source = np.zeros(self.size_source, dtype="int")
        z_source[us < proba_z_source] = 1

        a_target = np.zeros(self.dim_target)
        a_target[actives] = b_source

        y_target = np.dot(x_target, a_target)

        def M(x):
            return (
                sqrtm(cov_source)
                @ sqrtm(np.linalg.inv(cov_target))
                @ (x - self.mean_x_target)
                + self.mean_x_source
            )

        mx_target = np.array([M(x) for x in x_target])

        y_target = np.dot(mx_target, a_target) + np.random.normal(
            loc=0, scale=np.sqrt(sigma_source), size=self.size_target
        )

        y_target = (y_target - y_target.mean()) / y_target.std()
        y_source = (y_source - y_source.mean()) / y_source.std()

        a_target = np.zeros(self.dim_target)
        a_target[actives] = np.log(self.odds_ratio_target)

        z = np.dot(mx_target, a_target)

        proba_z_target = np.exp(z) / (1 + np.exp(z))

        us = np.random.uniform(0, 1, self.size_target)

        z_target = np.zeros(self.size_target, dtype="int")
        z_target[us < proba_z_target] = 1

        data_source["Y"] = y_source
        data_source["Z"] = z_source
        data_target["Y"] = y_target
        data_target["Z"] = z_target

        return data_source, data_target


class DataScenarioTest(DataScenario):
    """
    Sample a test sample for the reference scenario (300 observations)
    """

    def __init__(self):
        super().__init__()
        self.size_source = 300
        self.size_target = 300
        self.obs_covar_prop_source = 1.0
        self.obs_covar_prop_target = 1.0
