import math
import numpy as np
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))

from jdcoot import DataScenario, DataScenarioTest
from jdcoot.performance import compute

nsimulations = 100

train_size = 1000
test_size = 1000

train_scenario = DataScenario()
train_scenario.size_source = train_size
train_scenario.size_target = train_size
test_scenario = DataScenarioTest()
test_scenario.size_source = test_size
test_scenario.size_target = test_size

json_file = "odds_ratio_variations.json"
with open(json_file, "w+") as f:
    f.seek(0)

odds_ratio_values = [0.2, 0.4, 0.6, 0.8]

variable_types = ["discrete"]
learning_methods = ["unsupervised"]
recoding_methods = ["coot", "jdcoot", "reference"]

for i in range(nsimulations):
    indices = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)

    for odds_ratio in odds_ratio_values:
        train_scenario.odds_ratio_source = odds_ratio
        train_scenario.odds_ratio_target = odds_ratio
        test_scenario.odds_ratio_source = odds_ratio
        test_scenario.odds_ratio_target = odds_ratio

        compute(
            json_file,
            train_scenario,
            test_scenario,
            indices,
            variable_types,
            learning_methods,
            recoding_methods,
        )
