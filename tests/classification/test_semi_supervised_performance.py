import numpy as np

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.path.dirname(__file__),'..','..','src').resolve()))

print(sys.path)
from jdcoot import Sref_test, Sref, Sref_Poisson, Sref_Poisson_test
from jdcoot import data_generator, Performance
from jdcoot.globals import INDEX_GENERATION

# # Reference Scenario function

test_Data = Sref_test(INDEX_GENERATION)
reference_Data = Sref(INDEX_GENERATION)

Poisson_Data_Reference = Sref_Poisson(INDEX_GENERATION)
Poisson_Data_Test = Sref_Poisson_test(INDEX_GENERATION)

# # Test on Categorial artificial Data 

# ## Test of Performance function

# ### Independent covariables

# +
D = data_generator(n_Source=1000, n_Target=1000, d_Source=100, d_Target=100, mean_X_Source=np.zeros(100),
                   mean_X_Target=np.zeros(100), mean_Y_Source=0,
                   mean_Y_Target=0, Source_Generation_Correlation=0, Source_Non_Generation_Correlation=0,
                   Target_Generation_Correlation=0, Target_Non_Generation_Correlation=0, Sparse_Rate=0.75,
                   Odds_Ratio_Source=0.5, Odds_Ratio_Target=0.5, R2_Source=0.6, R2_Target=0.6,
                   Observed_Covariates_Proportion_Source=0.2, Observed_Covariates_Proportion_Target=0.2,
                   Indexes_Chosen_For_Generation=INDEX_GENERATION)

Performance(D, test_Data, type_supervision='semi-supervised', Objective_Variable='discrete', Balance=True, alpha=None,
            Labelled_Proportion_Target=0.2)
