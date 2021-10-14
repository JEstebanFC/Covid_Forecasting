#!/usr/bin/env python
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from optparse import OptionParser
from Models import DATA_PATH, RESULTS_PATH

from Models.Models import Models

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
    for state in options.state:
        if state not in states:
            print(state + ' is not available')
            continue
        activeCases(state)
        models = Models(state)
        rmse = {}
        for model in options_models:
            if model in arima_models:
                errors,pred = models.ARIMA(model)
            if model in regression_models:
                errors,pred = models.regression(model)
            rmse[model] = errors
        RMSE.loc[state] = rmse
    print()
    print(RMSE.to_string())