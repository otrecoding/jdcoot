import os
import sys
import math
import numpy as np
from sklearn.model_selection import train_test_split


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))
import jdcoot
from jdcoot import DataScenario, DataScenarioTest


np.random.seed(1972)

INDEX_GENERATION = np.random.choice(
    np.arange(100), math.ceil(0.75 * 100), replace=False
)

size = 1000

reference_scenario = DataScenario()
reference_scenario.size_source = size
reference_scenario.size_target = size
test_scenario = DataScenarioTest()
test_scenario.size_source = size // 5
test_scenario.size_target = size // 5

source, target = reference_scenario.generate(INDEX_GENERATION)

source_test, target_test = test_scenario.generate(INDEX_GENERATION)

source_test = source_test.loc[:, source.columns]
target_test = target_test.loc[:, target.columns]

prop_source = 0.1
prop_target = 0.1

n_target = len(target.Z)
n_source = len(source.Z)
l_source_train, l_source_test = train_test_split(
    np.arange(n_source), train_size=prop_source
)
l_target_train, l_target_test = train_test_split(
    np.arange(n_target), train_size=prop_target
)

pure_source, pure_target, test_source, test_target = (
    jdcoot.continuous_partial_reference(
        source,
        target,
        source_test,
        target_test,
        l_source_train,
        l_source_test,
        l_target_train,
        l_target_test,
    )
)

print(f"Pure performance on source : {pure_source} ")
print(f"Pure performance on target : {pure_target} ")
print(f"Test performance on source : {test_source} ")
print(f"Test performance on target : {test_target} ")
