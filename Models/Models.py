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
        self.dates = state_data['Date']
        daterange = self.dates.values
        date_index = pd.date_range(start=daterange[0], end=daterange[-1], freq='D')
        self.last_date = daterange[-1]
        self.activecases = pd.Series(data, date_index)
        totActiveCases = self.activecases.values.reshape(-1,1)

        state_data_reduce = state_data.iloc[:,[4,5,7,8,9,10,11,12,13,14,15,16,23]]

        data_quantity = state_data.shape[0]
        # 70% training data, 30 for validation
        train_ml = state_data.iloc[:int(data_quantity*0.70)]
        valid_ml = state_data.iloc[int(data_quantity*0.70):]
        train_dates = self.dates.iloc[:int(data_quantity*0.70)].values
        valid_dates = self.dates.iloc[int(data_quantity*0.70):].values
        self.trainActiveCases = totActiveCases[:int(data_quantity*0.70)]
        self.validActiveCases = totActiveCases[int(data_quantity*0.70):]
        #Regression
        self.ml_all_f = state_data_reduce.values
        self.train_ml_all_f = state_data_reduce.iloc[:int(data_quantity*0.70)].values
        self.valid_ml_all_f = state_data_reduce.iloc[int(data_quantity*0.70):].values
        #ARIMA
        train_index = pd.date_range(start=train_dates[0], periods=len(train_dates), freq='D')
        self.valid_index = pd.date_range(start=valid_dates[0], periods=len(valid_dates), freq='D')
        self.train_active = pd.Series(train_ml['Active Cases'].values, train_index)
        self.valid_active = pd.Series(valid_ml['Active Cases'].values, self.valid_index)

    def getDailyCases(self):
        covidDB = CovidDB()
        self.dailyCases = covidDB.newDailyCasesCountries(self.country)
    
    def __errors(self,activeCases,prediction):
        RMSE = np.sqrt(mse(activeCases,prediction))
        MAE = mae(activeCases,prediction)
        R2 = r2_score(activeCases,prediction)
        return [RMSE,MAE,R2]
    
    def __regression(self, regression):
        wn.filterwarnings("ignore")
        regression.fit(self.train_ml_all_f,self.trainActiveCases)
        wn.filterwarnings("default")
        prediction = regression.predict(self.valid_ml_all_f)
        errors = self.__errors(self.validActiveCases,prediction)
        pred = regression.predict(self.ml_all_f)
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

    def plotActiveCases(self):
        f, ax = plt.subplots(1,1, figsize=(12,10))
        plt.plot(self.activecases)
        title = 'Active case History for ' + self.country
        ax.set_title(title)
        ax.set_ylabel("No of Active Covid-19 Cases")
        ax.set_xlim([dt.date(2020, 3, 1), dt.date(2020, 5, 1)])
        plt.savefig(self.results_path + 'active_cases\\' + self.last_date + '_{country}_active_cases.png'.format(country=self.country))

    def plotRegression(self,prediction,model):
        plt.figure(figsize=(11,6))
        plt.plot(self.activecases.values,label="Active Cases")
        plt.plot(self.dates, prediction, linestyle='--',label="Predicted Active Cases using {model} Regression".format(model=model),color='black')
        plt.title("Active Cases {model} Regression Prediction".format(model=model))
        plt.xlabel('Time')
        plt.ylabel('Active Cases')
        plt.xticks(rotation=90)
        plt.legend()
        plt.savefig(self.results_path + 'regression\\' + self.last_date + '_{country}_{model}_regression.png'.format(country=self.country,model=model.lower()))

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
        plt.savefig(self.results_path + 'arima\\' + self.last_date + '_{country}_{model}.png'.format(country=self.country,model=model))
        # plot residual errors
        residuals.plot()
        resError = self.results_path + 'arima\\resError\\'
        plt.savefig(resError + self.last_date + '_{country}_{model}_residual_error.png'.format(country=self.country,model=model))
        residuals.plot(kind='kde')
        plt.savefig(resError + self.last_date + '_{country}_{model}_residual_error_kde.png'.format(country=self.country,model=model))
