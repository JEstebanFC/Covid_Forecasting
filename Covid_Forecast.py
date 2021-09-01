#!/usr/bin/env python
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from optparse import OptionParser
from Models import DATA_PATH, RESULTS_PATH

from Models.Arima import Arima
from Models.Varmax import Varmax
from Models.Fbprophet import Fbprophet
from Models.Smoothing import Smoothing
from Models.Regression import Regression
from Models.DickeyFuller import DickeyFuller


def activeCases(state):
    active_path = RESULTS_PATH + '\\active_cases\\'
    df_per_State_features = pd.read_csv(DATA_PATH + state +'.csv')
    df_per_State_features = df_per_State_features.fillna(0)
    df_per_State_features["Active Cases"].replace({0:1}, inplace=True)
    df_state_recs = df_per_State_features
    last_date = df_state_recs['Date'].values[-1]

    df_per_State_features = df_state_recs
    data = df_per_State_features['Active Cases'].astype('double').values
    daterange = df_per_State_features['Date'].values
    date_index = pd.date_range(start=daterange[0], end=daterange[-1], freq='D')
    activecases = pd.Series(data, date_index)

    f, ax = plt.subplots(1,1, figsize=(12,10))
    plt.plot(activecases)
    ax.set_ylabel("No of Active Covid-19 Cases")
    title = 'Active case History for ' + state
    ax.set_title(title)
    ax.set_xlim([datetime.date(2020, 3, 1), datetime.date(2020, 5, 1)])
    plt.savefig(active_path + last_date + '_{state}_active_cases.png'.format(state=state))

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-m', '--model', dest='model', default='')
    parser.add_option('-s', '--state', dest='state', default='list')
    options, args = parser.parse_args()

    # options.state = 'Maharashtra,Kerala,Delhi'
    # options.model = 'regression'

    states_list = open(DATA_PATH + 'States_list.csv', 'r')
    states = states_list.readline().split(',')
    states_list.close()
    
    models = ['regression','linear','polynomial','lasso','smoothing','exponential','holts']
    options.model = options.model.split(',')

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
    
    
    for state in options.state:
        if state not in states:
            print(state + ' is not available')
            continue
        try:
            activeCases(state)
            regression = Regression(state)
            if 'regression' in options.model or 'lasso' in options.model:
                regression.lassoRegression()
            if 'regression' in options.model or 'linear' in options.model:
                regression.linearRegression()
            if 'regression' in options.model or 'polynomial' in options.model:
                regression.polynomialRegression()
            if 'tslean' in options.model:
                regression.tslean()
            
            arima = Arima(state)
            if 'ARIMA' in options.model or 'ar' in options.model:
                arima.AR()
            if 'ARIMA' in options.model or 'ma' in options.model:
                arima.MA()
            if 'ARIMA' in options.model or 'arima' in options.model:
                arima.ARIMA()
            if 'ARIMA' in options.model or 'sarimax' in options.model:
                arima.SARIMAX()
            
            smoothing = Smoothing(state)
            if 'smoothing' in options.model or 'exponential' in options.model:
                smoothing.exponentialSmoothing()
            if 'smoothing' in options.model or 'holts' in options.model:
                smoothing.holtsWinter()
            
            if 'dickey' in options.model or 'fuller' in options.model:
                df = DickeyFuller(state)
                df.dickeyFuller()
            
            if 'fb' in options.model or 'fbprophet' in options.model or 'prophet' in options.model:
                fbprophet = Fbprophet(state)
                fbprophet.prophet()
            
            # Different results...
            varmax = Varmax(state)
            if 'varmax' in options.model:
                varmax.AR()
                varmax.VMA()
                varmax.ARIMA()

        except:
            print(state + ' failed')

