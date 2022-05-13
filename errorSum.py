#!/usr/bin/env python
import os
import pandas as pd
import numpy as np

from Models.Models import Models
# from Utils.CovidDB import CovidDB

plotResults = True
if plotResults:
    PATH = os.getcwd()

    folder = '98'
    ePath = PATH + '\\Results\\2022-04-29\\{folder}\\Errors\\'.format(folder=folder)
    rPath = PATH + '\\Results\\2022-04-29\\{folder}\\Results\\'.format(folder=folder)
    models = {}
    models['arima'] = 'ARIMA'
    models['prophet'] = 'Prophet'
    models['lstm'] = 'LSTM'

    summarize_results = False
    if summarize_results:
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
        for m in models:
            aErrors.replace(m,models[m],inplace=True)
        aErrors.reset_index(inplace=True)
        aErrors.set_index(['Dataset','Metric'],inplace=True)
        aErrors.to_csv(rPath + 'AllErrors.csv')
    else:
        index = ['04w', '08w', '10w', '20w', '30w', '40w', 'Full']
        aErrors = pd.read_csv(rPath + 'AllErrors.csv')
        aErrors.reset_index(inplace=True)
        aErrors.set_index(['Dataset','Metric'],inplace=True)

    wsm = aErrors.xs('WSM',level=1)
    countries = np.unique(wsm['Countries'].values)
    
    lineStyle = ['o-C0','o-C1','o-C2']
    Models.plots_path = rPath

    labels = ['Weeks','Metric']
    for c in countries:
        wsmc = wsm.loc[wsm['Countries']==c]
        xData = []
        yData = []
        legends = []
        fileName = '{country}.png'.format(country=c)
        for m in models:
            xData.append(wsmc[m].index)
            yData.append(wsmc[m].values)
            legends.append(m.upper())
            title = 'Metric for {country} over weeks'.format(country=c)
        Models.plot(Models,xData,yData,lineStyle,legends,labels,fileName,title)
    
    lineStyle = ['o-C0','o-C1','o-C2','o-C3','o-C4','o-C5']
    for m in models:
        xData = []
        yData = []
        legends = []
        fileName = '{model}.png'.format(model=m)
        for c in countries:
            wsmc = wsm.loc[wsm['Countries']==c]
            xData.append(wsmc[m].index)
            yData.append(wsmc[m].values)
            legends.append(c)
            title = '{Model} Metric for every country over weeks'.format(Model=m.upper())
        Models.plot(Models,xData,yData,lineStyle,legends,labels,fileName,title)
    
    labels = ['Country','Metric']
    ylim = [0,1.90]
    for i in index:
        xData = []
        yData = []
        legends = []
        fileName = '{index}.png'.format(index=i)
        for m in models:
            wsmi = wsm.loc[i]
            xData.append(wsmi['Countries'].values)
            yData.append(wsmi[m].values)
            legends.append(m)
            title = '{index} Metric for every country over weeks'.format(index=i)
        Models.plot(Models,xData,yData,lineStyle,legends,labels,fileName,title,ylim=ylim)


dailyPlots = False
if dailyPlots:
    countries = ['Australia','New Zealand','China','Japan','US','Germany']
    dataOpts = {}
    dataOpts['initDay'] = None
    dataOpts['lastDay'] = '2022-04-29'
    dataOpts['forecast'] = 0
    dataOpts['train_percent'] = '4w'
    dataOpts['plot'] = False

    for country in countries:
        models = Models(country)
        t = models.selectData(**dataOpts)

        #Plotting Daily Cases
        xData = [models.activecases.index]
        yData = [models.activecases.values]
        linestyle = ['o-C0']
        legends = ['Daily cases']
        labels = ['Date Time','Daily Cases']
        fileName = '{country}_daily_cases.png'.format(country=country)
        title = 'Daily case for ' + country

        vertical = []
        timeFrame = [10,20,30,40]
        for tf in timeFrame:
            last = models.activecases.index[-1] - pd.DateOffset(7*tf)
            vertical.append([last,str(tf)+'w',''])

        models.plot(xData, yData, linestyle, legends, labels, fileName, title, plotLimit=False, vertical=vertical)






