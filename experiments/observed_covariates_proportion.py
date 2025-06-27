import json
import math
import numpy as np
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
sys.path.append(os.path.abspath('src'))

from jdcoot import DataScenario, DataScenarioTest
from jdcoot.performance import models, compute


nsimulations = 100

train_scenario = DataScenario()
test_scenario = DataScenarioTest()
    
json_file = 'observed_covariates_proportion.json'
with open(json_file, 'w+') as f:
    f.seek(0)

variable_types = ["continuous", "discrete"]
learning_methods = ["unsupervised"]
recoding_methods = ["coot", "jdcoot", "reference"]

values = [0.2, 0.4, 0.6, 0.8]

for i in range(nsimulations):

    indices = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)
    
    for pxo_source in values:
        for pxo_target in values:

            train_scenario.obs_covar_prop_source = pxo_source
            train_scenario.obs_covar_prop_target = pxo_target

            compute(json_file, train_scenario, test_scenario, indices, 
                      variable_types, learning_methods, recoding_methods)
    

