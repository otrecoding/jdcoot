import os
import sys
sys.path.append(os.path.abspath('src'))
import jdcoot

data = jdcoot.generate_data()

proportions = [0.1, 0.5, 0.9]
for prop_source in proportions:
    for prop_target in proportions:
        print(f'prop_source: {prop_source} prop_target: {prop_target}')

        perf_source, perf_target = jdcoot.discrete_partial_coot( *data, prop_source = prop_source, prop_target = prop_target )

        print(f"Performance on source : {perf_source} ")
        print(f"Performance on target : {perf_target} ")
