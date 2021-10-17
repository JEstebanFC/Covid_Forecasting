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

class Models:
    def __init__(self, country):
        self.country = country
        self.results_path = RESULTS_PATH + 'IT819\\'

        state_data = pd.read_csv(DATA_PATH + self.country + '.csv')
        state_data = state_data.fillna(0)
        state_data["Active Cases"].replace({0:1}, inplace=True)
        data = state_data['Active Cases'].astype('double').values
        data_quantity = state_data.shape[0]

        self.dates = state_data['Date']
        date_index = pd.date_range(start=self.dates.values[0], end=self.dates.values[-1], freq='D')
        state_data["Days Since"] = date_index - date_index[0]
        state_data["Days Since"] = state_data["Days Since"].dt.days
        self.daysSince = state_data["Days Since"]

        self.activecases = pd.Series(data, date_index)
        totActiveCases = self.activecases.values.reshape(-1,1)
        self.trainActiveCases = totActiveCases[:int(data_quantity*0.70)]
        self.validActiveCases = totActiveCases[int(data_quantity*0.70):]
        
        train_ml = state_data.iloc[:int(data_quantity*0.70)]
        valid_ml = state_data.iloc[int(data_quantity*0.70):]
        self.train_index = self.daysSince[:int(data_quantity*0.70)]
        self.valid_index = self.daysSince[int(data_quantity*0.70):]
        self.train_active = pd.Series(train_ml['Active Cases'].values, self.train_index)
        self.valid_active = pd.Series(valid_ml['Active Cases'].values, self.valid_index)

    def getDailyCases(self):
        covidDB = CovidDB()
        self.dailyCases = covidDB.dailyCases(self.country)
        covidDB.plotDailyCases(self.country)
    
    def __errors(self,validCases,prediction):
        RMSE = np.sqrt(mse(validCases,prediction))
        MAE = mae(validCases,prediction)
        R2 = r2_score(validCases,prediction)
        return [RMSE,MAE,R2]
    
    def __regression(self, regression):
        wn.filterwarnings("ignore")
        regression.fit(self.train_index.values.reshape(-1,1),self.trainActiveCases)
        wn.filterwarnings("default")
        prediction = regression.predict(self.valid_index.values.reshape(-1,1))
        errors = self.__errors(self.validActiveCases,prediction)
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
        self.plotRegression(pred, method)
        return errors, pred

    def __ARIMA(self, model):
        model_fit = model.fit()
        prediction = model_fit.forecast(len(self.validActiveCases))
        errors = self.__errors(self.validActiveCases,prediction)
        pred = pd.Series(prediction, self.valid_index)
        residuals = pd.DataFrame(model_fit.resid)
        return errors, pred, residuals

    def ARIMA(self, method):
        method = method.upper()
        if method == 'AR':
            model = ARIMA(self.trainActiveCases, order=(2, 0, 0))
        elif method == 'MA':
            model = ARIMA(self.trainActiveCases, order=(0, 0, 2))
        elif method == 'ARIMA':
            model = ARIMA(self.trainActiveCases, order=(1, 1, 1))
        errors,pred,residuals = self.__ARIMA(model)
        self.plotARIMA(pred,residuals,method)
        return errors, pred

    def plotRegression(self,prediction,model):
        plt.figure(figsize=(11,6))
        plt.plot(self.activecases, label="Active Cases")
        plt.plot(self.activecases.index, prediction, linestyle='--',label="Predicted Active Cases using {model} Regression".format(model=model),color='black')
        plt.title("Active Cases {model} Regression Prediction".format(model=model))
        plt.xlabel('Time')
        plt.ylabel('Active Cases')
        # plt.xticks(rotation=90)
        plt.legend()
        plt.savefig(self.results_path + 'regression\\' + self.dates.values[-1] + '_{country}_{model}_regression.png'.format(country=self.country,model=model.lower()))

    def plotARIMA(self,prediction,residuals,model):
        # Plotting
        f, ax = plt.subplots(1,1 , figsize=(12,10))
        plt.plot(self.train_active, marker='o',color='blue',label ="Train Data Set")
        plt.plot(self.valid_active, marker='o',color='green',label ="Valid Data Set")
        plt.plot(prediction, marker='o',color='red',label ="Predicted " + model)
        plt.legend()
        plt.xlabel("Date Time")
        plt.ylabel('Active Cases')
        plt.title("Active Cases {model} Model Forecasting for {country}".format(country=self.country,model=model))
        plt.savefig(self.results_path + 'arima\\' + self.dates.values[-1] + '_{country}_{model}.png'.format(country=self.country,model=model))
        # plot residual errors
        residuals.plot()
        resError = self.results_path + 'arima\\resError\\'
        plt.savefig(resError + self.dates.values[-1] + '_{country}_{model}_residual_error.png'.format(country=self.country,model=model))
        residuals.plot(kind='kde')
        plt.savefig(resError + self.dates.values[-1] + '_{country}_{model}_residual_error_kde.png'.format(country=self.country,model=model))
