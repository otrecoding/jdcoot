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

train_size = 1000
test_size = 1000 // 4

train_scenario = DataScenario()
train_scenario.size_source = train_size
train_scenario.size_target = train_size
test_scenario = DataScenarioTest()
test_scenario.size_source = test_size
test_scenario.size_target = test_size
    
json_file = 'r2_variations.json'
with open(json_file, 'w+') as f:
    f.seek(0)

r2_values = [0.2, 0.4, 0.6, 0.8]

variable_types = ["continuous"]
learning_methods = ["unsupervised"]
recoding_methods = ["coot", "jdcoot", "train"]

for i in range(nsimulations):
    
    indices = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)

    for r2_source in r2_values:
        #for r2_target in r2_values:

            train_scenario.r2_source = r2_source
            train_scenario.r2_target = r2_source
            test_scenario.r2_source = r2_source
            test_scenario.r2_target = r2_source

            compute(json_file, train_scenario, test_scenario, indices, 
                    variable_types, learning_methods, recoding_methods)
    

