import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))
import jdcoot

data = jdcoot.generate_data()
pure_source, pure_target, test_source, test_target = jdcoot.continuous_partial_coot(
    *data, prop_source=0.1, prop_target=0.2
)

print(f"Pure performance on source : {pure_source} ")
print(f"Pure performance on target : {pure_target} ")
print(f"Test performance on source : {test_source} ")
print(f"Test performance on target : {test_target} ")
