import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))
import jdcoot

data = jdcoot.generate_data()
perf_source, perf_target = jdcoot.continuous_partial_coot(
    *data, prop_source=0.1, prop_target=0.2
)

print(f"Performance COOT on source : {perf_source} ")
print(f"Performance COOT on target : {perf_target} ")
