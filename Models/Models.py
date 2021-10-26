import os
import numpy as np
from numpy.testing._private.utils import print_assert_equal
import pandas as pd
import datetime as dt
import warnings as wn
import statsmodels.api as sm
import matplotlib.pyplot as plt

from Utils.CovidDB import CovidDB

from Models import DATA_PATH, RESULTS_PATH

from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.arima.model import ARIMA

class Models:
    def __init__(self, country):
        self.country = country
        self.covidDB = CovidDB()
        
    def selectData(self, initDay=None, lastDay=None, forecast=0, train_percent=0.7):
        self.activecases,self.daysSince = self.getDailyCases(country=self.country, initDay=initDay, lastDay=lastDay)
        if self.activecases.empty:
            return
        lastDay = self.daysSince.values[-1]
        forecast_index = pd.date_range(start=self.daysSince.index[-1], periods=forecast+1, freq='D')[1:]
        self.forecastDays = pd.Series(range(lastDay+1, lastDay+1+forecast), forecast_index)

        train_quantity = int(self.activecases.shape[0]*train_percent)
        train_ml = self.activecases[:train_quantity]
        valid_ml = self.activecases[train_quantity:]
        self.train_index = self.daysSince[:train_quantity]
        self.valid_index = self.daysSince[train_quantity:]
        self.train_active = train_ml.values.reshape(-1,1)
        self.valid_active = valid_ml.values.reshape(-1,1)

    def createFolders(self, firstDay, lastDay):
        new_folder = '{lastDay}\\{firstDay}'.format(lastDay=lastDay,firstDay=firstDay)
        self.plots_path = self.covidDB.createFolder(new_folder)
        self.csv_path = self.covidDB.createFolder(new_folder + '\\csv')
        self.resPath = self.covidDB.createFolder(new_folder + '\\residual')

    def getDailyCases(self, country=None, initDay=None, lastDay=None):
        if not country:
            country = self.country
        dailyCases = self.covidDB.dailyCases(country)
        if dailyCases.empty:
            print('Error: No data found for the country selected')
            return pd.Series(),pd.Series()
        try:
            dailyCases = dailyCases.loc[country].loc['']
        except:
            dailyCases = dailyCases.loc[country].sum()
        
        if not lastDay or lastDay not in dailyCases:        #Assuming there is information for every day inside the range
            lastDay = dailyCases.index[-1]
        if initDay != None and 'w' in initDay:
            weeks = int(initDay.split('w')[0])
            initDay = lastDay - pd.DateOffset(7*weeks)
        if initDay == None or initDay not in dailyCases:    #Assuming there is information for every day inside the range
            initDay = dailyCases.index[0]
        dailyCases = dailyCases[initDay:lastDay]
        casesFrame = dailyCases.to_frame('Daily Cases')
        casesFrame["Days Since"] = casesFrame.index - casesFrame.index[0]
        casesFrame["Days Since"] = casesFrame["Days Since"].dt.days
        daysSince = pd.Series(casesFrame['Days Since'].values, casesFrame.index)

        self.createFolders(firstDay=str(initDay.date()), lastDay=str(lastDay.date()))
        casesFrame.to_csv(self.csv_path + '{country}_daily_cases.csv'.format(country=country))
        # Plotting Daily Cases
        xData = [dailyCases.index]
        yData = [dailyCases.values]
        linestyle = ['-C0']
        legends = ['Daily cases']
        labels = ['Date Time','Daily Cases']
        fileName = '{country}_daily_cases.png'.format(country=country)
        title = 'Daily case for ' + country
        self.plot(xData, yData, linestyle, legends, labels, fileName, title)

        return dailyCases,daysSince

    def __errors(self,validCases,prediction):
        RMSE = np.sqrt(mse(validCases,prediction))
        MAE = mae(validCases,prediction)
        R2 = r2_score(validCases,prediction)
        return [RMSE,MAE,R2]
    
    def __regression(self, method, poly=3):
        X_values = self.train_index.values.reshape(-1,1)
        if method.lower() == 'linear':
            regression = LinearRegression(normalize=True)
        elif method.lower() == 'lasso':
            lasso_reg = Lasso(alpha=.8,normalize=True, max_iter=1e5)
            regression = make_pipeline(PolynomialFeatures(3), lasso_reg)
        elif 'polynomial' in method.lower():
            t = method.lower().split('polynomial')[-1]
            if t:
                poly = int(t)
            regression = make_pipeline(PolynomialFeatures(degree=poly), LinearRegression())
            # regression = make_pipeline(PolynomialFeatures(degree=poly), Ridge())
        regression.fit(X_values,self.train_active)
        prediction = regression.predict(self.valid_index.values.reshape(-1,1))
        errors = self.__errors(self.valid_active,prediction)
        pred = regression.predict(self.daysSince.append(self.forecastDays).values.reshape(-1,1))
        return errors,pred

    def regression(self, method):
        method = method.capitalize()
        errors,pred = self.__regression(method)
        # Plotting
        xData = [self.activecases.index,self.activecases.index.append(self.forecastDays.index)]
        yData = [self.activecases.values,pred]
        vertical = [self.valid_index.index[0]]
        if not self.forecastDays.empty:
            vertical.append(self.forecastDays.index[0])
        linestyle = ['-C0','-r']
        legends = ['Daily Cases',"Predicted Daily Cases: {model} Regression".format(model=method)]
        labels = ['Date Time','Daily Cases']
        fileName = '{country}_regression_{model}.png'.format(country=self.country, model=method.lower())
        title = "Daily Cases {model} Regression Prediction for {country}".format(country=self.country,model=method)
        self.plot(xData, yData, linestyle, legends, labels, fileName, title, vertical=vertical)
        return errors

    def __ARIMA(self, order):
        preds = []
        history = [x for x in self.train_active]
        for val in self.valid_active:
            model = ARIMA(history, order=order)
            model_fit = model.fit()
            prediction = model_fit.forecast()
            preds.append(prediction[0])
            history.append(val)
            # history.append(np.array([prediction[0]]))
        errors = self.__errors(self.valid_active,preds)
        forecast = []
        for fore in self.forecastDays:
            model = ARIMA(history, order=order)
            model_fit = model.fit()
            prediction = model_fit.forecast()
            forecast.append(prediction[0])
            history.append(np.array([prediction[0]]))
        # if not self.forecastDays.empty:
        #     forecast = model_fit.forecast(len(self.forecastDays))
        forecast = pd.Series(forecast, self.forecastDays.index)
        pred = pd.Series(preds, self.valid_index)
        residuals = pd.DataFrame(model_fit.resid)
        return errors, pred, forecast, residuals

    def ARIMA(self, method):
        method = method.upper()
        if method == 'AR':
            order = (2, 0, 0)
        elif method == 'MA':
            order = (0, 0, 2)
        elif method == 'ARIMA':
            order = (5, 1, 0)
            # order = (1, 1, 1)
        errors,pred,forecast,residuals = self.__ARIMA(order)
        # Plotting
        xData = [self.train_index.index, self.valid_index.index, self.valid_index.index]
        yData = [self.train_active, self.valid_active, pred]
        vertical = [self.valid_index.index[0]]
        linestyle = ['o-b','o-g','o-r']
        legends = ['Train Data Set', 'Valid Data Set', 'Predicted '+ method + str(order).replace(' ','')]
        if len(forecast) != 0:
            xData.append(self.forecastDays.index)
            yData.append(forecast)
            linestyle.append('*-k')
            legends.append('{days} of Forecast'.format(days=len(self.forecastDays)))
            vertical.append(self.forecastDays.index[0])
        labels = ['Date Time','Daily Cases']
        fileName = '{country}_{model}.png'.format(country=self.country, model=method + str(order))
        title = "Daily Cases {model} Model Forecasting for {country}".format(country=self.country,model=method)
        self.plot(xData, yData, linestyle, legends, labels, fileName, title, vertical=vertical)
        self.plotResidualsARIMA(residuals, method)
        return errors

    def plot(self, xData, yData, lineStyle, legends, labels, fileName, title, **opts):
        plt.figure(figsize=(12,10))
        plt.title(title)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.xticks(rotation=45)
        for xd,yd,ls,l in zip(xData, yData, lineStyle, legends):
            plt.plot(xd, yd, ls, label=l)
        if 'vertical' in opts:
            for ver in opts['vertical']:
                plt.axvline(x=ver, color='k', linestyle='--')
        plt.legend(loc=2)
        plt.savefig(self.plots_path + fileName)

    def plotResidualsARIMA(self, residuals, model):
        residuals.plot()
        plt.savefig(self.resPath + '{country}_{model}_residual_error.png'.format(country=self.country, model=model))
        residuals.plot(kind='kde')
        plt.savefig(self.resPath + '{country}_{model}_residual_error_kde.png'.format(country=self.country, model=model))
