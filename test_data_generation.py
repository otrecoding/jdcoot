import os, sys
sys.path.append(os.path.abspath('src'))
import math
import numpy as np
import pandas as pd

import jdcoot

INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)

S, T = jdcoot.Sref(INDEX_GENERATION)
S_test, T_test = jdcoot.Sref_test(INDEX_GENERATION)

print(S.columns)
print(S_test.columns)

reference_scenario = jdcoot.DataScenario()
test_scenario = jdcoot.DataScenarioTest()

source, target = reference_scenario.generate(INDEX_GENERATION)
test_source, test_target = test_scenario.generate(INDEX_GENERATION)


print(source.columns)
print(test_source.columns)

