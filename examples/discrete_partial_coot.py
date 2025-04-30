import os
import sys
sys.path.append(os.path.abspath('src'))
import importlib
import jdcoot
importlib.reload(jdcoot.models.discrete_partial_coot)
from jdcoot.models.discrete_partial_coot import discrete_partial_coot

if __name__ == "__main__":

    from jdcoot.scenario import generate_data

    data = generate_data()


    proportions = [0.1, 0.5, 0.9]
    for prop_source in proportions:
        for prop_target in proportions:
            print(f'prop_source: {prop_source} prop_target: {prop_target}')

            perf_pure, perf_test = jdcoot.discrete_partial_coot( *data, prop_source = prop_source, prop_target = prop_target )
    
            print(f"Pure Performance COOT : {perf_pure} ")
            print(f"Test Performance COOT : {perf_test} ")
