import numpy as np
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))

import jdcoot

source, target, source_test, target_test = jdcoot.generate_data()

l_source_train, l_source_test = None, None
l_target_train, l_target_test = None, None

pure_source, pure_target, test_source, test_target = jdcoot.discrete_unsupervised_coot(
    source,
    target,
    source_test,
    target_test,
    l_source_train,
    l_source_test,
    l_target_train,
    l_target_test,
)

print(f"Pure performance on source : {pure_source} ")
print(f"Pure performance on target : {pure_target} ")
print(f"Test performance on source : {test_source} ")
print(f"Test performance on target : {test_target} ")
