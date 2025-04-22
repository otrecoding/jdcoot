import os
import sys
sys.path.append(os.path.abspath('src'))
import jdcoot

if __name__ == "__main__":

    from jdcoot.scenario import generate_data

    data = generate_data()

    perf_pure, perf_test = jdcoot.discrete_partial_coot( *data )
    
    print(f"Pure Performance COOT : {perf_pure} ")
    print(f"Test Performance COOT : {perf_test} ")
