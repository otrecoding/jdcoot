import os
import sys

sys.path.append(os.path.abspath("src"))
import jdcoot

data = jdcoot.generate_data()

proportions = [0.1, 0.5, 0.9]
for prop_source in proportions:
    for prop_target in proportions:
        print(f"prop_source: {prop_source} prop_target: {prop_target}")

        pure_source, pure_target, test_source, test_target = jdcoot.discrete_partial_coot(
            *data, prop_source=prop_source, prop_target=prop_target
        )

        print(f"Pure performance on source : {pure_source} ")
        print(f"Pure performance on target : {pure_target} ")
        print(f"Test performance on source : {test_source} ")
        print(f"Test performance on target : {test_target} ")
