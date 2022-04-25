#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt

from optparse import OptionParser
from Models import RESULTS_PATH

from Models.Models import Models
from numpy import argsort

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
    parser.add_option('-r', '--ratio', dest='ratio', type=str, default='0.75')
    parser.add_option('--daily-plot', dest='dailyPlot', action='store_true', default=False)
    options, args = parser.parse_args()
    options.countries = options.countries.split(',')
    opts_models = []
    for om in options.model.split(','):
        opts_models.append(om.lower())
    
    options_models = []
    for model in opts_models:
        if model in options_models:
            continue
        if model in ['arima','lstm', 'prophet']:
            options_models.append(model)
        # elif 'polynomial' in model:
        #     options_models.append(model)

    dataOpts = {}
    dataOpts['initDay'] = options.firstDay
    dataOpts['lastDay'] = options.lastDay
    dataOpts['forecast'] = options.prediction
    dataOpts['train_percent'] = options.ratio
    dataOpts['plot'] = options.dailyPlot
    
    resultsPath = []

    metrics = ['MAE', 'RMSE', 'R2', 'MAPE', 'NRMSE', 'WSM']
    metrics = ['R2', 'MAPE', 'NRMSE', 'WSM']
    errorMetrics = {}
    for e in metrics:
        errorMetrics[e] = pd.DataFrame(columns=options_models, index=options.countries)
        errorMetrics[e].index.name = 'Countries'
        for country in options.countries:
            errorMetrics[e].loc[country] = {}
    
    for country in options.countries:
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
            if model == 'arima':
                errors,pred,forecast = models.ARIMA()
            if model == 'lstm':
                errors,pred,forecast = models.LSTM()
            if model == 'prophet':
                errors,pred,forecast = models.prophet()
            for e in metrics:
                errorMetrics[e].loc[country][model] = errors[e]

    if options_models != []:
        for e in metrics:
            print('\n' + color.BOLD + e + color.END)
            print(errorMetrics[e].to_string())
            print()
            for country in options.countries:
                metricDict = dict(errorMetrics[e].loc[country])
                metrics = list(metricDict.keys())
                if e == 'R2':
                    errors = 1 - pd.array(list(metricDict.values()))
                else:
                    errors = pd.array(list(metricDict.values()))
                ranking = ''
                for i in argsort(errors):
                    if ranking:
                        ranking += ', '
                    ranking += metrics[i]
                print('Ranking for %s: %s' %(country,ranking))

    print(color.BOLD + '\nResults saved in: ' + color.END)
    for p in resultsPath:
        print('\t',p)
    print()

    errors = pd.concat(list(errorMetrics.values()),keys=list(errorMetrics.keys()),axis=0)
    errors.to_csv(resultsPath[0] + 'csv\\Errors.csv')



