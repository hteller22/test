
import pandas as pd
import numpy as np
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

def posibilities(opt_list):
    """
    Generates all possible combinations of a list of options, from zero to the length of the input list.

    Parameters:
    opt_list (list): A list of options.

    Returns:
    list: A list of lists, representing all possible combinations of the input options.
    """
    # Initialize an empty list to hold all possible combinations
    list_posibilities=[]
    
    # Iterate over all possible subset sizes of the input list
    for L in range(0, len(opt_list)+1):
        # Generate all possible combinations of the input list for the current subset size
        for subset in itertools.combinations(opt_list, L):
            # Append the current combination to the list of all possible combinations
            list_posibilities.append(list(subset))
    
    # Filter out any empty lists and return the final list of possible combinations
    return list(filter(None, list_posibilities))