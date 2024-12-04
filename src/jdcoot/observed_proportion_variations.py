import numpy as np

from .cross_variation import Cross_Partial_Labelled_Proportion_Variations
from .data_observation import Semi_Supervised_Labelled_Proportion_Variations
from .proportion_variations import Partial_Labelled_Proportion_Variations

# +


def Observed_Labels_Proportions_Variation(Data, Data_test=None, type_supervision='unsupervised',
                                          Proportion_Labelled_Variations=np.array([0.1, 0.5, 0.9]), Monte_Carlo=1,
                                          algo='both', Objective_Variable='discrete', Multi_Variations=True,
                                          Balance=False, alpha=None):
    """
    Observed_Labels_Proportions_Variation : Compute and plot performance versus the proportion of labelled observations in target and/or source. Permit to choose what type of supervision

    Input : Data (tuple or str) : Learning data (tuple of 2), tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                            | X_1 | ... | X_d | Y | Z |

                                        with X_i the ith observed covariate,
                                             Y the continuous objective variable for regression analysis (optional),
                                             Z the discrete objective variable for classification analysis (optional)

                                OR

                                (str) Name of the folder where data has been stored

                                OR

                                Previous simulated data (tuple of 10) with respectively :
                                                            Pure performance Monte Carlo COOT,
                                                             Pure performance Monte Carlo JDCOOT,
                                                             Pure performance Monte Carlo REF,
                                                             Pure performance Boxplot COOT,
                                                             Pure performance Boxplot JDCOOT,
                                                             Test performance Monte Carlo COOT,
                                                             Test performance Monte Carlo JDCOOT,
                                                             Test performance Monte Carlo REF,
                                                             Test performance Boxplot COOT,
                                                             Test performance Boxplot JDCOOT

            Data_test (tuple) : Test data, tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                            | X_1 | ... | X_d | Y | Z |

                                        with X_i the ith observed covariate,
                                             Y the continuous objective variable for regression analysis (optional),
                                             Z the discrete objective variable for classification analysis (optional)

                                If None, Learning data is splited into 30% of test data and 70% of learning data


            type_supervision (str : 'unsupervised' or 'semi-supervised' or 'partial' or 'cross-partial') : if 'unsupervised', measure the performance such that none of the observations of target are labelled and all source observations are labelled
                                                                                                           if 'semi-supervised', measure the performance such that labelled proportions of observations of target vary
                                                                                                           if 'partial', measure the performance such that labelled proportions of observations of target and source vary
                                                                                                           if 'cross-partial', measure the performance such that labelled proportions of observations of target and source vary. Here, for JDCOOT, uses 2 semi-supervised JDCOOT methods for each dataset


            Proportion_Labelled_Variations : np array of values of labelled proportions to vary

            Monte_Carlo : Number of repetitions of computing


            Multi_Variations (bool) : if true, compute performance along source and target simultaneous variation of proportions of labelled observations, and plot heatmaps


            Objective_Variable (str : 'discrete' or 'continuous') : if 'discrete', measure the performance for classifisation
                                                                    if 'continuous', measure the performance for regression

            algo (str : 'COOT' or 'JDCOOT' or 'both') : if 'COOT', measure the performance for COOT
                                                        if 'JDCOOT', measure the performance for JDCOOT
                                                        if 'both',  measure the performance for COOT and JDCOOT

            Balance (bool) : if true, balance the number of observation in each class. So, source and target dscrete variables have the same distribution. However, it destroys some of the data.

            alpha (float) : Hyper parameter of the problem formulation as in the paper. If None, optimized alpha is used (optimized for data given by the previous function data_generator())





    Output : Data (tuple) : tuple of 10 or 14 elements with respectively : For Multivariation Scenarios (Heatmap) : 14 elements :
                                                                            Pure performance Monte Carlo COOT,
                                                                            Pure performance Monte Carlo JDCOOT,
                                                                            Pure performance Monte Carlo REF,
                                                                            Pure performance Boxplot COOT,
                                                                            Pure performance Boxplot JDCOOT,
                                                                            Pure performance Heatmap COOT,
                                                                            Pure performance Heatmap JDCOOT,
                                                                            Test performance Monte Carlo COOT,
                                                                            Test performance Monte Carlo JDCOOT,
                                                                            Test performance Monte Carlo REF,
                                                                            Test performance Boxplot COOT,
                                                                            Test performance Boxplot JDCOOT,
                                                                            Test performance Heatmap COOT,
                                                                            Test performance Heatmap JDCOOT,



                                                                        For Univariation Scenarios (No Heatmap) : 10 elements :
                                                                            Pure performance Monte Carlo COOT,
                                                                            Pure performance Monte Carlo JDCOOT,
                                                                            Pure performance Monte Carlo REF,
                                                                            Pure performance Boxplot COOT,
                                                                            Pure performance Boxplot JDCOOT,
                                                                            Test performance Monte Carlo COOT,
                                                                            Test performance Monte Carlo JDCOOT,
                                                                            Test performance Monte Carlo REF,
                                                                            Test performance Boxplot COOT,
                                                                            Test performance Boxplot JDCOOT

    """

    if type_supervision == 'partial':
        return Partial_Labelled_Proportion_Variations(Data, Data_test, Proportion_Labelled_Variations, Monte_Carlo,
                                                      algo, Objective_Variable, Multi_Variations, Balance, alpha)

    elif type_supervision == 'unsupervised':
        return Semi_Supervised_Labelled_Proportion_Variations(Data, Data_test, np.array([0]), Monte_Carlo, algo,
                                                              Objective_Variable, Balance, alpha)

    elif type_supervision == 'semi-supervised':
        return Semi_Supervised_Labelled_Proportion_Variations(Data, Data_test, Proportion_Labelled_Variations,
                                                              Monte_Carlo, algo, Objective_Variable, Balance, alpha)

    elif type_supervision == 'cross-partial':
        return Cross_Partial_Labelled_Proportion_Variations(Data, Data_test, Proportion_Labelled_Variations,
                                                            Monte_Carlo, algo, Objective_Variable, Multi_Variations,
                                                            Balance, alpha)
