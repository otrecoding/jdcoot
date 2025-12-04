import math
import numpy as np
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
sys.path.append(os.path.abspath('src'))
from jdcoot import DataScenario, DataScenarioTest
from jdcoot.performance import compute

nsimulations = 100

prop_source, prop_target = 0.2, 0.2
sizes = [10, 100, 500, 1000] 

train_scenario = DataScenario()
test_scenario = DataScenarioTest()

json_file = 'sample_size_variation.json'
with open(json_file, 'w+') as f:
    f.seek(0)

variable_types = ["continuous", "discrete"]
learning_methods = ["unsupervised"]
recoding_methods = ["coot", "jdcoot", "reference"]

sparse_rate = 75

for i in range(nsimulations):
    
    indices = np.random.choice(np.arange(100), math.ceil(sparse_rate), replace=False)

    for size in sizes:

        train_scenario.size_source = 1000
        train_scenario.size_target = size
        test_scenario.size_source = 1000
        test_scenario.size_target = size
        
        compute(json_file, train_scenario, test_scenario, indices, 
                variable_types, learning_methods, recoding_methods)
    

