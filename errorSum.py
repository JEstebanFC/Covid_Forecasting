#!/usr/bin/env python
import os
import pandas as pd

PATH = os.getcwd()

ePath = PATH + '\\Results\\2022-04-29\\1st\\Errors\\'
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

aErrors.to_csv(ePath + '..\\Results\\AllErrors.csv')