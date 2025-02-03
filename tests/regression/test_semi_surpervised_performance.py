# BASICS
import numpy as np  # scientific computing
import pandas as pd
import math
from jdcoot import data_generator
from jdcoot import Performance
from jdcoot import Sref_test, Sref

# # Data Generation

# ### Same variables for generation, to stay in the "same world"

global INDEX_GENERATION
INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)

# # Reference Scenario function

test_Data = Sref_test(INDEX_GENERATION)
reference_Data = Sref(INDEX_GENERATION)

# # Test on Continuous artificial Data 

# ## Test of Performance function

# ### Independent covariables

D = data_generator(n_Source=1000, n_Target=1000, d_Source=100, d_Target=100, mean_X_Source=np.zeros(100),
                   mean_X_Target=np.zeros(100), mean_Y_Source=0,
                   mean_Y_Target=0, Source_Generation_Correlation=0, Source_Non_Generation_Correlation=0,
                   Target_Generation_Correlation=0, Target_Non_Generation_Correlation=0, Sparse_Rate=0.75,
                   Odds_Ratio_Source=0.5, Odds_Ratio_Target=0.5, R2_Source=0.6, R2_Target=0.6,
                   Observed_Covariates_Proportion_Source=0.2, Observed_Covariates_Proportion_Target=0.2,
                   Indexes_Chosen_For_Generation=INDEX_GENERATION)

Performance(D, test_Data, type_supervision='semi-supervised', Objective_Variable='continuous', Balance=False,
            alpha=None, Labelled_Proportion_Target=0.2)
