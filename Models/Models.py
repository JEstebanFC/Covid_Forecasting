import os
import numpy as np
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
    def __init__(self, country, initDay=None, lastDay=None):
        self.country = country
        self.covidDB = CovidDB()
        self.selectData(initDay=initDay, lastDay=lastDay)
        if not lastDay:
            lastDay = str(self.daysSince.index[-1].date())
        self.results_path = self.covidDB.createFolder(lastDay)
        self.resPath = self.covidDB.createFolder(lastDay + '\\residual\\')

    def selectData(self, train_percent=0.7, initDay=None, lastDay=None):
        state_data = self.getDailyCases()

        date_index = pd.date_range(start=state_data['Date'].values[0], periods=len(state_data['Date']), freq='D')
        state_data["Days Since"] = date_index - date_index[0]
        state_data["Days Since"] = state_data["Days Since"].dt.days
        self.daysSince = pd.Series(state_data['Days Since'].values, date_index)
        self.activecases = pd.Series(state_data['Active Cases'].values, date_index)

        train_quantity = int(state_data.shape[0]*train_percent)
        train_ml = self.activecases[:train_quantity]
        valid_ml = self.activecases[train_quantity:]
        self.train_index = self.daysSince[:train_quantity]
        self.valid_index = self.daysSince[train_quantity:]
        self.train_active = train_ml.values.reshape(-1,1)
        self.valid_active = valid_ml.values.reshape(-1,1)

    def getDailyCases(self, country=None):
        if not country:
            country = self.country
        if country in ['Maharashtra','Delhi']:
            dailyCases = pd.read_csv(DATA_PATH + country + '.csv')
        else:
            self.covidDB.plotDailyCases(country)
            dailyCases = self.covidDB.dailyCases(country)
            try:
                dailyCases = dailyCases.loc[country].loc['']
            except:
                dailyCases = dailyCases.loc[country].sum()
            dailyCases.index.name = 'Date'
            dailyCases = dailyCases.to_frame('Active Cases')
            dailyCases = dailyCases.reset_index()
        dailyCases.fillna(0)
        # dailyCases["Active Cases"].replace({0:1}, inplace=True)
        return dailyCases

    def __errors(self,validCases,prediction):
        RMSE = np.sqrt(mse(validCases,prediction))
        MAE = mae(validCases,prediction)
        R2 = r2_score(validCases,prediction)
        return [RMSE,MAE,R2]
    
    def __regression(self, regression):
        wn.filterwarnings("ignore")
        regression.fit(self.train_index.values.reshape(-1,1),self.train_active)
        wn.filterwarnings("default")
        prediction = regression.predict(self.valid_index.values.reshape(-1,1))
        errors = self.__errors(self.valid_active,prediction)
        pred = regression.predict(self.daysSince.values.reshape(-1,1))
        return errors,pred

    def regression(self, method):
        method = method.capitalize()
        if method.lower() == 'linear':
            regression = LinearRegression(normalize=True)
        elif method.lower() == 'polynomial':
            regression = PolynomialFeatures(degree = 7) 
            regression = make_pipeline(PolynomialFeatures(3), Ridge())
        elif method.lower() == 'lasso':
            lasso_reg = Lasso(alpha=.8,normalize=True, max_iter=1e5)
            regression = make_pipeline(PolynomialFeatures(3), lasso_reg)
        errors,pred = self.__regression(regression)
        # Plotting
        xData = [self.activecases.index,self.activecases.index]
        yData = [self.activecases.values,pred]
        linestyle = ['-C0','--k']
        legends = ['Active Cases',"Predicted Active Cases: {model} Regression".format(model=method)]
        labels = ['Date Time','Active Cases']
        fileName = '{country}_regression_{model}.png'.format(country=self.country, model=method.lower())
        title = "Active Cases {model} Regression Prediction for {country}".format(country=self.country,model=method)
        self.plot(xData, yData, linestyle, legends, labels, fileName, title)
        return errors, pred

    def __ARIMA(self, model):
        model_fit = model.fit()
        prediction = model_fit.forecast(len(self.valid_active))
        errors = self.__errors(self.valid_active,prediction)
        pred = pd.Series(prediction, self.valid_index)
        residuals = pd.DataFrame(model_fit.resid)
        return errors, pred, residuals

    def ARIMA(self, method):
        method = method.upper()
        if method == 'AR':
            order = (2, 0, 0)
        elif method == 'MA':
            order = (0, 0, 2)
        elif method == 'ARIMA':
            order = (1, 1, 1)
        model = ARIMA(self.train_active, order=order)
        errors,pred,residuals = self.__ARIMA(model)
        # Plotting
        xData = [self.train_index.index, self.valid_index.index, self.valid_index.index]
        yData = [self.train_active, self.valid_active, pred]
        linestyle = ['o-b','o-g','o-r']
        legends = ['Train Data Set', 'Valid Data Set', 'Predicted '+ method]
        labels = ['Date Time','Active Cases']
        fileName = '{country}_{model}.png'.format(country=self.country,model=method)
        title = "Active Cases {model} Model Forecasting for {country}".format(country=self.country,model=method)
        self.plot(xData, yData, linestyle, legends, labels, fileName, title)
        self.plotResidualsARIMA(residuals, method)
        return errors, pred

    def plot(self, xData, yData, lineStyle, legends, labels, fileName, title):
        plt.figure(figsize=(12,10))
        plt.title(title)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.xticks(rotation=45)
        for xd,yd,ls,l in zip(xData, yData, lineStyle, legends):
            plt.plot(xd, yd, ls, label=l)
        plt.legend(loc=2)
        plt.savefig(self.results_path + fileName)

    def plotResidualsARIMA(self, residuals, model):
        residuals.plot()
        plt.savefig(self.resPath + '{country}_{model}_residual_error.png'.format(country=self.country,model=model))
        residuals.plot(kind='kde')
        plt.savefig(self.resPath + '{country}_{model}_residual_error_kde.png'.format(country=self.country,model=model))
