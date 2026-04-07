import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))
import jdcoot
from jdcoot.scenario import generate_data

data  = generate_data()

source, target, source_test, target_test = data

prop_source=0.95
prop_target=0.005
y_target = target.Y.values[:, np.newaxis]
n_target = len(y_target)
y_source = source.Y.values[:, np.newaxis]
n_source = len(y_source)

l_source_train, l_source_test = train_test_split(
   np.arange(n_source), train_size=prop_source
)
l_target_train, l_target_test = train_test_split(
    np.arange(n_target), train_size=prop_target
)

pure_source, pure_target, test_source, test_target = jdcoot.discrete_semisupervised_reference(
        *data, l_source_train, l_source_test, l_target_train, l_target_test
)

print(f"Pure performance COOT on source : {pure_source} ")
print(f"Pure performance COOT on target : {pure_target} ")
print(f"Test performance COOT on source : {test_source} ")
print(f"Test performance COOT on target : {test_target} ")
