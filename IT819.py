#!/usr/bin/env python
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from optparse import OptionParser
from Models import DATA_PATH, RESULTS_PATH

from Models.Models import Models

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-m', '--model', dest='model', default='')
    parser.add_option('-s', '--state', dest='state', default='list')
    options, args = parser.parse_args()
    opts_models = []
    for om in options.model.split(','):
        opts_models.append(om.lower())
    
    states_list = open(DATA_PATH + 'States_list.csv', 'r')
    states = states_list.readline().split(',')
    states_list.close()
    #### Active cases ####
    if options.state.lower() == 'list':
        for st in states:
            print(st)
        exit()
    if options.state.lower() == 'all':
        options.state = []
        for st in states:
            options.state.append(st)
    else:
        options.state = options.state.split(',')
    
    options_models = []
    arima_models = ['ar','ma','arima']
    regression_models = ['linear','polynomial','lasso']
    if 'arimas' in opts_models:
        options_models.extend(arima_models)
    if 'regression' in opts_models:
        options_models.extend(regression_models)
    for model in opts_models:
        if model in options_models:
            continue
        if model in regression_models or model in arima_models:
            options_models.append(model)

    RMSE = pd.DataFrame(columns=options_models, index=options.state)
    RMSE.index.name = 'States'
    MAE = pd.DataFrame(columns=options_models, index=options.state)
    MAE.index.name = 'States'
    R2 = pd.DataFrame(columns=options_models, index=options.state)
    R2.index.name = 'States'
    for state in options.state:
        if state not in states:
            print(state + ' is not available')
            continue
        # try:
        rmse = {}
        mae = {}
        r2 = {}
        models = Models(state)
        models.plotActiveCases()
        for model in options_models:
            if model in arima_models:
                errors,pred = models.ARIMA(model)
            if model in regression_models:
                errors,pred = models.regression(model)
            rmse[model] = errors[0]
            mae[model] = errors[1]
            r2[model] = errors[2]
        RMSE.loc[state] = rmse
        MAE.loc[state] = mae
        R2.loc[state] = r2
        # except:
        #     print(state + ' failed')
    print('\nRMSE')
    print(RMSE.to_string())
    print('\nMAE')
    print(MAE.to_string())
    print('\nR2')
    print(R2.to_string())
    print()