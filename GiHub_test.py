
import pandas as pd
import numpy as np
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

def posibilities(opt_list):
    list_posibilities=[]
    for L in range(0, len(opt_list)+1):
        for subset in itertools.combinations(opt_list, L):
            list_posibilities.append(list(subset))
    return list(filter(None, list_posibilities))