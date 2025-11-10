import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
sys.path.append(os.path.abspath('src'))
from jdcoot import DataScenario, DataScenarioTest
from jdcoot.performance import models, compute

indices = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)

nsimulations = 100

train_size = 1000
test_size = 1000 

train_scenario = DataScenario()
train_scenario.size_source = train_size
train_scenario.size_target = train_size
test_scenario = DataScenarioTest()
test_scenario.size_source = test_size
test_scenario.size_target = test_size
    
json_file = 'observed_labels_proportions_variation.json'
with open(json_file, 'w+') as f:
    f.seek(0)

variable_types = ["continuous", "discrete"]
learning_methods = ["reference", "unsupervised", "semisupervised", "partial"]
recoding_methods = ["coot", "jdcoot", "reference"]

prop_source_values = [0.001, 0.005, 0.01, 0.05, 0.1]
prop_target_values = [0.001, 0.005, 0.01, 0.05, 0.1]

for i in range(nsimulations):
    compute(json_file, train_scenario, test_scenario, indices, 
            variable_types, ["unsupervised"], recoding_methods)
    for prop_target in prop_target_values:
        compute(json_file, train_scenario, test_scenario, indices, 
                variable_types, ["semisupervised"], recoding_methods, prop_target = prop_target)
        for prop_source in prop_source_values:
            compute(json_file, train_scenario, test_scenario, indices, 
                    variable_types, ["reference", "partial"], recoding_methods,
                    prop_source = prop_source, prop_target = prop_target)
