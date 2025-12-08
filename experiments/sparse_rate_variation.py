import math
import numpy as np
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))

from jdcoot import DataScenario, DataScenarioTest
from jdcoot.performance import compute

nsimulations = 100

prop_source, prop_target = 0.2, 0.2
size = 1000
size_source_train = size
size_target_train = size
size_source_test = size
size_target_test = size

test_scenario = DataScenarioTest()
train_scenario = DataScenario()

train_scenario.size_source = size_source_train
train_scenario.size_target = size_target_train
test_scenario.size_source = size_source_test
test_scenario.size_target = size_target_test

json_file = "sparse_rate_variation.json"
with open(json_file, "w+") as f:
    f.seek(0)

variable_types = ["continuous", "discrete"]
learning_methods = ["unsupervised"]
recoding_methods = ["coot", "jdcoot"]

for i in range(nsimulations):
    for sparse_rate in [0.25, 0.5, 0.75, 1]:
        indices = np.random.choice(
            np.arange(100), math.ceil(sparse_rate * 100), replace=False
        )
        compute(
            json_file,
            train_scenario,
            test_scenario,
            indices,
            variable_types,
            learning_methods,
            recoding_methods,
        )
