import os
import sys
import math
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
sys.path.append(os.path.abspath('src'))
import jdcoot
from jdcoot import DataScenario, DataScenarioTest


np.random.seed(1972)

INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)

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

perf_pure, perf_test = jdcoot.continuous_partial_reference(source, target, source_test, target_test)

print(f"Pure Performance Reference : {perf_pure}")
print(f"Test Performance Reference : {perf_test}")
