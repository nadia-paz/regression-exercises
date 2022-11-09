import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm

import wrangle as wr
import explore as ex

def regression_errors(y, yhat):
    '''
    this function accepts actual results and predictions
    as array or pd.Series and calculates
    evaluation scores
    returns SSE, TSS, ESS, MSE, RMSE scores
    '''
    # calculate predictions' residuals
    residual = yhat - y
    # calculate baseline's residuals
    residual_baseline = y - y.mean()

    # sum of squared errors score
    SSE = (residual ** 2).sum()
    
    # total sum of squares score
    TSS = (residual_baseline ** 2).sum()
    
    # explained sum of squares score
    ESS = TSS - SSE
    
    # mean squared error score
    MSE = SSE/len(y)
    
    # root mean squared error score
    RMSE = MSE ** .5
    
    return SSE, TSS, ESS, MSE, RMSE

def baseline_mean_errors(y):
    '''
    this function accepts a y_train array or pd.Series
    calculates baseline error scores
    returns SSE_baseline, MSE_baseline, RMSE_baseline scores
    '''
    # calculate baseline's residuals
    residual_baseline = y - y.mean()

    # sum of squared errors score for baseline
    SSE_baseline = (residual_baseline ** 2).sum()

    # mean squared error score for baseline
    MSE_baseline = SSE_baseline / len(y)

    # root mean squared error score for baseline
    RMSE_baseline = MSE_baseline ** .5
    
    return SSE_baseline, MSE_baseline, RMSE_baseline

def better_than_baseline(y, yhat):
    '''
    this function accepts actual results and predictions
    as arrays or pd.Series, calls other functions to 
    obtain SSE scores for the predictions and baseline
    prints whether model or baseline are better
    '''
    SSE, _, _, _, _ = regression_errors(y, yhat)
    SSE_baseline, _, _ = baseline_mean_errors(y)
    
    difference = SSE - SSE_baseline
    if difference > 0:
        print('Baseline is better')
    else:
        print('Model is better')