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

prop_source, prop_target = 0.2, 0.2
sizes = [10, 100, 500, 1000] 
test_size = 300
test_scenario = DataScenarioTest()
train_scenario = DataScenario()

json_file = 'sparse_rate_variation.json'
with open(json_file, 'w+') as f:
    f.seek(0)

variable_types = ["continuous", "discrete"]
learning_methods = ["unsupervised"]
recoding_methods = ["coot", "jdcoot", "reference"]

for i in range(nsimulations):
    
    indices = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)

    size_source_train = 1000
    size_target_train = 1000
    size_source_test = size_source_train // 3
    size_target_test = size_target_train // 3

    test_scenario.size_source = size_source_test
    test_scenario.size_target = size_target_test

    train_scenario.size_source = size_source_train
    train_scenario.size_target = size_target_train

    for sparse_rate in [0.25, 0.5, 0.75, 1]:

        train_scenario.sparse_rate = sparse_rate
        test_scenario.sparse_rate = sparse_rate

        compute(json_file, train_scenario, test_scenario, indices, 
                variable_types, learning_methods, recoding_methods)
    

