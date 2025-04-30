import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
sys.path.append(os.path.abspath('src'))
import numpy as np
import jdcoot

if __name__ == "__main__":
    data = jdcoot.generate_data()
    pure, test = jdcoot.continuous_partial_coot(*data, prop_source = 0.1, prop_target = 0.2)

    print(f"Pure Performance COOT : {pure} ")
    print(f"Test Performance COOT : {test} ")
