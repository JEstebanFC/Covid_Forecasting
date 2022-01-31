#!/usr/bin/env python
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from optparse import OptionParser
from Models import RESULTS_PATH

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
    parser.add_option('-r', '--ratio', dest='ratio', type=float, default=0.7)
    parser.add_option('--daily-plot', dest='dailyPlot', action='store_true', default=False)
    options, args = parser.parse_args()
    options.countries = options.countries.split(',')
    opts_models = []
    for om in options.model.split(','):
        opts_models.append(om.lower())
    
    options_models = []
    # arima_orders = [(2,0,0), (0,0,2), (5,1,0), (1,1,1)]
    # arima_orders = [(5,1,0), (1,1,1)]
    # arima_orders = [(5,1,0), (5,1,1),(5,2,1), (5,2,2)]
    arima_orders = ['ARIMA']
    # regression_models = ['linear','polynomial','lasso']
    regression_models = ['linear','lasso']
    if 'arima' in opts_models or 'arimas' in opts_models:
        options_models.extend(arima_orders)
    if 'regression' in opts_models:
        options_models.extend(regression_models)
    for model in opts_models:
        if model in options_models:
            continue
        if model in regression_models or model in arima_orders or model == 'lstm':
            options_models.append(model)
        elif 'polynomial' in model:
            options_models.append(model)

    dataOpts = {}
    dataOpts['initDay'] = options.firstDay
    dataOpts['lastDay'] = options.lastDay
    dataOpts['forecast'] = options.prediction
    dataOpts['train_percent'] = options.ratio
    dataOpts['plot'] = options.dailyPlot
    RMSE = pd.DataFrame(columns=options_models, index=options.countries)
    RMSE.index.name = 'Countries'
    MAE = pd.DataFrame(columns=options_models, index=options.countries)
    MAE.index.name = 'Countries'
    R2 = pd.DataFrame(columns=options_models, index=options.countries)
    R2.index.name = 'Countries'
    resultsPath = []
    for country in options.countries:
        rmse = {}
        mae = {}
        r2 = {}
        models = Models(country=country)
        t = models.selectData(**dataOpts)
        if t.empty:
            print('Error: No data found for ' + country)
            continue
        p = models.plots_path
        if p not in resultsPath:
            resultsPath.append(p)
        for model in options_models:
            print('Starting with {country} using {model} model'.format(country=country,model=model))
            if model in arima_orders:
                errors,pred,forecast = models.ARIMA(model)
            if model in regression_models or 'polynomial' in model:
                errors,pred = models.regression(model)
            if model == 'lstm':
                errors,pred,forecast = models.LSTM()
            rmse[model] = errors[0]
            mae[model] = errors[1]
            r2[model] = errors[2]
        RMSE.loc[country] = rmse
        MAE.loc[country] = mae
        R2.loc[country] = r2
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
    errors = pd.concat([RMSE,MAE,R2],keys=['RMSE','MAE','R2'],axis=0)
    errors.to_csv(resultsPath[0] + 'csv\\Errors.csv')
    # print(errors)

