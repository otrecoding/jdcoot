jdcoot documentation
====================

Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.

Import and Utils
----------------

.. code-block:: python

   import os, sys
   sys.path.append(os.path.abspath('src'))
   import math
   import numpy as np  # scientific computing
   import pandas as pd
   from jdcoot import *

Data Generation
---------------

Same variables for generation, to stay in the "same world"

.. code-block:: python

   global INDEX_GENERATION
   INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)

Reference Scenario function
---------------------------

.. code-block:: python

   test_Data = Sref_test(INDEX_GENERATION)
   reference_Data = Sref(INDEX_GENERATION)

Test on Continuous artificial Data 
----------------------------------

Test of Performance function
----------------------------

Independent covariables

.. code-block:: python

   D = data_generator(n_Source=1000, n_Target=1000, d_Source=100, d_Target=100, mean_X_Source=np.zeros(100),
                      mean_X_Target=np.zeros(100), mean_Y_Source=0,
                      mean_Y_Target=0, Source_Generation_Correlation=0, Source_Non_Generation_Correlation=0,
                      Target_Generation_Correlation=0, Target_Non_Generation_Correlation=0, Sparse_Rate=0.75,
                      Odds_Ratio_Source=0.5, Odds_Ratio_Target=0.5, R2_Source=0.6, R2_Target=0.6,
                      Observed_Covariates_Proportion_Source=0.2, Observed_Covariates_Proportion_Target=0.2,
                      Indexes_Chosen_For_Generation=INDEX_GENERATION)
   Performance(D, test_Data, type_supervision='semi-supervised', Objective_Variable='continuous', Balance=False,
               alpha=None, Labelled_Proportion_Target=0.2)

Auto Correlated covariables 

.. code-block:: python

   Performance(reference_Data, test_Data, type_supervision='unsupervised', Objective_Variable='continuous', Balance=False,
            alpha=None, Labelled_Proportion_Target=0.2)

Data Observation variations
---------------------------

Learning cases impact

Unsupervised

.. code-block:: python

   UN = Observed_Labels_Proportions_Variation(reference_Data, test_Data, 'unsupervised', Monte_Carlo=10,
                                           Objective_Variable='continuous', algo='both', Balance=False, alpha=None)

Semi-supervised
---------------

.. code-block:: python

   SE = Observed_Labels_Proportions_Variation(reference_Data, test_Data, 'semi-supervised',
                                              np.array([0, 0.02, 0.05, 0.07, 0.1, 0.12, 0.15]), Monte_Carlo=10,
                                              Objective_Variable='continuous', algo='both', Balance=False, alpha=None)
   
   SE = Observed_Labels_Proportions_Variation(reference_Data, test_Data, 'semi-supervised',
                                              np.array([0.1, 0.3, 0.5, 0.7, 0.9]), Monte_Carlo=10,
                                              Objective_Variable='continuous', algo='both', Balance=False, alpha=None)

Partial and Cross-Partial
-------------------------

.. code-block:: python

   PA = Observed_Labels_Proportions_Variation(reference_Data, test_Data, 'partial',
                                              Proportion_Labelled_Variations=np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
                                              Monte_Carlo=5, algo='both', Objective_Variable='continuous',
                                              Multi_Variations=True, Balance=False, alpha=None)
   
   PA = Observed_Labels_Proportions_Variation(reference_Data, test_Data, 'partial',
                                              Proportion_Labelled_Variations=np.array([0.02, 0.05, 0.07, 0.1]),
                                              Monte_Carlo=5, algo='both', Objective_Variable='continuous',
                                              Multi_Variations=True, Balance=False, alpha=None)
   
   PA = Observed_Labels_Proportions_Variation(reference_Data, test_Data, 'cross-partial',
                                              Proportion_Labelled_Variations=np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
                                              Monte_Carlo=5, algo='both', Objective_Variable='continuous',
                                              Multi_Variations=True, Balance=False, alpha=None)
   
   PA = Observed_Labels_Proportions_Variation(reference_Data, test_Data, 'cross-partial',
                                              Proportion_Labelled_Variations=np.array([0.02, 0.05, 0.07, 0.1]),
                                              Monte_Carlo=5, algo='both', Objective_Variable='continuous',
                                              Multi_Variations=True, Balance=False, alpha=None)

Proportion of observed covariables variations

.. code-block:: python

   PXO = Proportion_Of_Observed_Covariates_Variation(Observed_Covariates_Proportion=np.array([0.2, 0.4, 0.6, 0.8]),
                                                  Monte_Carlo=10, Objective_Variable='continuous',
                                                  Multi_Variations=True, algo='both', type_supervision='unsupervised',
                                                  Balance=False, alpha=None)

Data Generation variations
--------------------------

Sample size variations

.. code-block:: python

   SSV = Sample_Size_Variation(Sample_sizes=np.array([10, 100, 500, 1000]), Monte_Carlo=10,
                            Objective_Variable='continuous', Multi_Variations=True, algo='both',
                            type_supervision='unsupervised', Balance=False, alpha=None)

:math:`R^2` variations

.. code-block:: python

   R2 = R2_Variation(R2=np.array([0.2, 0.4, 0.6, 0.8]), Monte_Carlo=10, Objective_Variable='continuous',
                  Multi_Variations=True, algo='both', type_supervision='unsupervised', Balance=False, alpha=None)

Sparse rate variations

.. code-block:: python

   SR = Sparse_Rate_Variation(SR=np.array([0.25, 0.5, 0.75, 1]), Objective_Variable='continuous', Monte_Carlo=10,
                           algo='both', type_supervision='unsupervised', Balance=False, alpha=None)

Mean shift

.. code-block:: python

   MS = Mean_Shift_Variation(Mean_Shift=np.array([0, 0.1, 0.2, 0.3, 0.4]), Objective_Variable='continuous', Monte_Carlo=10,
                          Multi_Variations=True, algo='both', type_supervision='unsupervised', Balance=False,
                          alpha=None)

Correlation variations

.. code-block:: python

   C = Correlation_Variation(Correlation=np.array([0, 0.2, 0.5, 0.7, 1]), Objective_Variable='continuous', Monte_Carlo=10,
                          Multi_Variations=True, algo='both', type_supervision='unsupervised', Balance=False,
                          alpha=None)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   classif
   comp
   coot
   correlation_variation
   covariate_variations
   cross_variation
   data_generation
   data_generation_variation
   data_observation
   graph_multi
   jdcot
   krr_classifier
   losses
   mean_shift_variation
   observed_proportion_variations
   performance
   proportion_variations
   qr_variation
   r2_variations
   read_data
   reference_scenario
   sinkhorn
   sparse_rate_variation
   store_data
   svn_classifier

