import re
import pandas as pd
import numpy as np

def get_distribution(values, thresholds):
    ''' Return a list of percentages of how the values are distributed among the given thresholds.
    Requires: 
      1) values: an array of values, for example an array of velocities or angles
      2) thresholds: an array of thresholds, for example: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    Returns:
      A list with how much percentage of the values falls under each given threshold. 
      For example: [< 0.1 %, 0.1 - 0.2 %, ... , 0.8 - 0.9 %, > 0.9%]
    '''
    percentages = []
    values = values[~np.isnan(values)]
    try:
        percentages.append(np.count_nonzero(np.asarray(values) < thresholds[0])/np.count_nonzero(~np.isnan(values)))
    except:
        percentages.append(0)
    for i in range(len(thresholds)-1):
        try:
            percentages.append(np.count_nonzero((np.asarray(values) >= thresholds[i])&(np.asarray(values) < thresholds[i+1]))/np.count_nonzero(~np.isnan(values)))
        except:
            percentages.append(0)
    try:
        percentages.append(np.count_nonzero(np.asarray(values) > thresholds[-1:])/np.count_nonzero(~np.isnan(values)))
    except:
        percentages.append(0)
    return percentages


def get_per_part(values, amount, func):
    ''' Perform a given function on N parts of the data
    Requires: 
      1) values
      2) amount: the amount of parts the values should be split into, this will also determine how many values are returned
      3) func: the function that will be applied to the values, for example: np.nanmedian
    Returns:
      A list containing the results after applying the given function on the given amount of parts. 
      For example: when amount is 4 and func is np.median, values will be slit in 4 parts 
                   and the median of each of these parts is returned.
    '''
    result = []
    length = np.count_nonzero(~np.isnan(values))
    part = int(round(length/amount))
    for i in range(amount):
        result.append(func(values[i*part:(i+1)*part]))
    return result