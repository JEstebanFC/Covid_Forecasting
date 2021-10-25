#!/usr/bin/env python
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from optparse import OptionParser
from Models import DATA_PATH, RESULTS_PATH

from Models.Models import Models

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-m', '--model', dest='model', default='')
    parser.add_option('-c', '--countries', dest='countries', default='')
    parser.add_option('-f', '--first-day', dest='firstDay', default=None)
    parser.add_option('-l', '--last-day', dest='lastDay', default=None)
    parser.add_option('-p', '--prediction', dest='prediction', type=int, default=0)
    options, args = parser.parse_args()
    options.countries = options.countries.split(',')
    opts_models = []
    for om in options.model.split(','):
        opts_models.append(om.lower())
    
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
        elif 'polynomial' in model:
            options_models.append(model)

    RMSE = pd.DataFrame(columns=options_models, index=options.countries)
    RMSE.index.name = 'Countries'
    MAE = pd.DataFrame(columns=options_models, index=options.countries)
    MAE.index.name = 'Countries'
    R2 = pd.DataFrame(columns=options_models, index=options.countries)
    R2.index.name = 'Countries'
    resultsPath = []
    for state in options.countries:
        rmse = {}
        mae = {}
        r2 = {}
        models = Models(country=state, forecast=options.prediction, initDay=options.firstDay, lastDay=options.lastDay)
        p = models.plots_path
        if p not in resultsPath:
            resultsPath.append(p)
        for model in options_models:
            if model in arima_models:
                errors = models.ARIMA(model)
            if model in regression_models or 'polynomial' in model:
                errors = models.regression(model)
            rmse[model] = errors[0]
            mae[model] = errors[1]
            r2[model] = errors[2]
        RMSE.loc[state] = rmse
        MAE.loc[state] = mae
        R2.loc[state] = r2
    print(color.BOLD + '\nRMSE' + color.END)
    print(RMSE.to_string())
    print(color.BOLD + '\nMAE' + color.END)
    print(MAE.to_string())
    print(color.BOLD + '\nR2' + color.END)
    print(R2.to_string())
    print(color.BOLD + '\nResults saved in: ' + color.END)
    for p in resultsPath:
        print('\t',p)
    print()
