#!/usr/bin/env python
import os
import pandas as pd
import numpy as np

from Models import Models

PATH = os.getcwd()

ePath = PATH + '\\Results\\2022-04-29\\1st\\Errors\\'
rPath = PATH + '\\Results\\2022-04-29\\1st\\Results\\'
eFiles = os.listdir(ePath)
index = ','.join(eFiles).replace('.csv','').split(',')

dfs = []
for ef in eFiles:
    df = pd.read_csv(ePath + ef)
    df.rename(columns={'Unnamed: 0': 'Metric'},inplace=True)
    df.set_index(['Metric'],inplace=True)
    dfs.append(df.round(decimals=5))

aErrors = pd.concat(dfs, keys=index)
aErrors.index.names = ['Dataset','Metric']
aErrors.replace('arima','ARIMA',inplace=True)
aErrors.replace('lstm','LSTM',inplace=True)
aErrors.replace('prophet','Prophet',inplace=True)
aErrors.reset_index(inplace=True)
aErrors.set_index(['Dataset','Metric'],inplace=True)

aErrors.to_csv(rPath + 'AllErrors.csv')

models = ['arima', 'prophet', 'lstm']
labels = ['Weeks','Metric']
lineStyle = ['o-C0']
Models.Models.plots_path = rPath

wsm = aErrors.xs('WSM',level=1)
countries = np.unique(wsm['Countries'].values)
for c in countries:
    wsmc = wsm.loc[wsm['Countries']==c]
    for m in models:
        xData = [wsmc[m].index]
        yData = [wsmc[m].values]
        legend = m.upper()
        fileName = '{country}_{model}.png'.format(country=c, model=m.upper())
        title = '{model} metric for {country} over weeks'.format(country=c,model=m.upper())
        Models.Models.plots_path = rPath
        Models.Models.plot(Models.Models,xData,yData,lineStyle,legend,labels,fileName,title)
        
  



