import math
import numpy as np
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))

from jdcoot import DataScenario, DataScenarioTest
from jdcoot.performance import compute

nsimulations = 100

train_scenario = DataScenario()
test_scenario = DataScenarioTest()

json_file = "correlation_variation.json"
with open(json_file, "w+") as f:
    f.seek(0)

coef_source_values = [0.2]
coef_target_values = [0.1, 0.2, 0.5, 0.7, 0.9]

variable_types = ["continuous", "discrete"]
learning_methods = ["unsupervised"]
recoding_methods = ["coot", "jdcoot", "reference"]

for i in range(nsimulations):
    indices = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)

    for coef_source in coef_source_values:
        for coef_target in coef_target_values:
            print(f"coef source = {coef_source} coef_target = {coef_target}")
            train_scenario.active_autocorr_source = coef_source
            train_scenario.active_autocorr_target = coef_target

            test_scenario.active_autocorr_source = coef_source
            test_scenario.active_autocorr_target = coef_target

            compute(
                json_file,
                train_scenario,
                test_scenario,
                indices,
                variable_types,
                learning_methods,
                recoding_methods,
            )
