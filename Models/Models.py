import numpy as np
import pandas as pd
import datetime as dt
import warnings as wn
import statsmodels.api as sm
import matplotlib.pyplot as plt

from Utils.CovidDB import CovidDB

from Models import DATA_PATH,DATA_PATH_NEW, RESULTS_PATH

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
    def __init__(self, state):
        self.state = state
        self.arima_path = RESULTS_PATH + '\\IT819\\arima\\'
        self.active_path = RESULTS_PATH + '\\IT819\\active_cases\\'
        self.regression_path = RESULTS_PATH + '\\IT819\\regression\\'

        self.df_per_State_features = pd.read_csv(DATA_PATH + self.state + '.csv')
        self.df_per_State_features = self.df_per_State_features.fillna(0)
        self.df_per_State_features["Active Cases"].replace({0:1}, inplace=True)

        data = self.df_per_State_features['Active Cases'].astype('double').values
        daterange = self.df_per_State_features['Date'].values
        self.last_date = daterange[-1]
        date_index = pd.date_range(start=daterange[0], end=daterange[-1], freq='D')
        self.activecases = pd.Series(data, date_index)
        self.totActiveCases = self.activecases.values.reshape(-1,1)

        df_per_State_sel_features = self.df_per_State_features.copy(deep=False)
        df_per_State_sel_features["Days Since"] = date_index - date_index[0]
        df_per_State_sel_features["Days Since"] = df_per_State_sel_features["Days Since"].dt.days
        df_per_State_sel_features = df_per_State_sel_features.iloc[:,[4,5, 7,8,9,10,11,12,13,14,15,16,23]]
        self.df_per_State_features["Days Since"] = date_index - date_index[0]
        self.df_per_State_features["Days Since"] = self.df_per_State_features["Days Since"].dt.days

        # 70% training data, 30 for validation
        self.train_ml = self.df_per_State_features.iloc[:int(self.df_per_State_features.shape[0]*0.70)]
        self.valid_ml = self.df_per_State_features.iloc[int(self.df_per_State_features.shape[0]*0.70):]
        self.train_dates = self.df_per_State_features['Date'].iloc[:int(df_per_State_sel_features.shape[0]*0.70)].values
        self.valid_dates = self.df_per_State_features['Date'].iloc[int(df_per_State_sel_features.shape[0]*0.70):].values
        self.trainActiveCases = self.totActiveCases[:int(self.df_per_State_features.shape[0]*0.70)]
        self.validActiveCases = self.totActiveCases[int(self.df_per_State_features.shape[0]*0.70):]
        #Regression
        self.ml_all_f = df_per_State_sel_features.values
        self.train_ml_all_f = df_per_State_sel_features.iloc[:int(df_per_State_sel_features.shape[0]*0.70)].values
        self.valid_ml_all_f = df_per_State_sel_features.iloc[int(df_per_State_sel_features.shape[0]*0.70):].values
        #ARIMA
        self.train_index = pd.date_range(start=self.train_dates[0], periods=len(self.train_dates), freq='D')
        self.valid_index = pd.date_range(start=self.valid_dates[0], periods=len(self.valid_dates), freq='D')
        self.train_active = pd.Series(self.train_ml['Active Cases'].values, self.train_index)
        self.valid_active = pd.Series(self.valid_ml['Active Cases'].values, self.valid_index)

    def getDailyCases(self):
        covidDB = CovidDB()
        self.dailyCases = covidDB.newCasesReport(self.state)
    
    def plotActiveCases(self):
        f, ax = plt.subplots(1,1, figsize=(12,10))
        plt.plot(self.activecases)
        title = 'Active case History for ' + self.state
        ax.set_title(title)
        ax.set_ylabel("No of Active Covid-19 Cases")
        ax.set_xlim([dt.date(2020, 3, 1), dt.date(2020, 5, 1)])
        plt.savefig(self.active_path + self.last_date + '_{state}_active_cases.png'.format(state=self.state))

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

    def plotRegression(self,prediction,model):
        plt.figure(figsize=(11,6))
        plt.plot(self.totActiveCases,label="Active Cases")
        plt.plot(self.df_per_State_features['Date'], prediction, linestyle='--',label="Predicted Active Cases using {model} Regression".format(model=model),color='black')
        plt.title("Active Cases {model} Regression Prediction".format(model=model))
        plt.xlabel('Time')
        plt.ylabel('Active Cases')
        plt.xticks(rotation=90)
        plt.legend()
        plt.savefig(self.regression_path + self.last_date + '_{state}_{model}_regression.png'.format(state=self.state,model=model.lower()))

    def plotARIMA(self,prediction,residuals,model):
        # Plotting
        f, ax = plt.subplots(1,1 , figsize=(12,10))
        plt.plot(self.train_active, marker='o',color='blue',label ="Train Data Set")
        plt.plot(self.valid_active, marker='o',color='green',label ="Valid Data Set")
        plt.plot(prediction, marker='o',color='red',label ="Predicted " + model)
        plt.legend()
        plt.xlabel("Date Time")
        plt.ylabel('Active Cases')
        plt.title("Active Cases {model} Model Forecasting for state {state}".format(state=self.state,model=model))
        plt.savefig(self.arima_path + self.last_date + '_{state}_{model}.png'.format(state=self.state,model=model))
        # plot residual errors
        residuals.plot()
        resError = self.arima_path + '\\resError\\'
        plt.savefig(resError + self.last_date + '_{state}_{model}_residual_error.png'.format(state=self.state,model=model))
        residuals.plot(kind='kde')
        plt.savefig(resError + self.last_date + '_{state}_{model}_residual_error_kde.png'.format(state=self.state,model=model))
