import numpy as np
import pandas as pd
import warnings as wn
import statsmodels.api as sm
import matplotlib.pyplot as plt

from Models import DATA_PATH,DATA_PATH_NEW, RESULTS_PATH

from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
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
        self.regression_path = RESULTS_PATH + '\\IT819\\regression\\'

        self.df_per_State_features = pd.read_csv(DATA_PATH + state +'.csv')
        self.df_per_State_features = self.df_per_State_features.fillna(0)
        self.df_per_State_features["Active Cases"].replace({0:1}, inplace=True)

        data = self.df_per_State_features['Active Cases'].astype('double').values
        daterange = self.df_per_State_features['Date'].values
        self.last_date = daterange[-1]
        date_index = pd.date_range(start=daterange[0], end=daterange[-1], freq='D')
        activecases = pd.Series(data, date_index)
        self.totActiveCases = activecases.values.reshape(-1,1)

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

    def __regression(self, regression):
        wn.filterwarnings("ignore")
        regression.fit(self.train_ml_all_f,self.trainActiveCases)
        wn.filterwarnings("default")
        poly_pred = regression.predict(self.valid_ml_all_f)
        RMSE = np.sqrt(mean_squared_error(self.validActiveCases,poly_pred))
        pred = regression.predict(self.ml_all_f)
        return RMSE,pred

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
        RMSE,pred = self.__regression(regression)
        print(color.BOLD + "\t{state}: Root Mean Square Error for {method} Regression: {RMSE}".format(state=self.state, method=method, RMSE=str(RMSE)) + color.END)
        self.plotRegression(pred, method)
        return RMSE, pred

    def __ARIMA(self, model):
        model_fit = model.fit()
        prediction = model_fit.forecast(len(self.validActiveCases))
        RMSE = np.sqrt(mean_squared_error(self.validActiveCases,prediction))
        pred_active = pd.Series(prediction, self.valid_index)
        residuals = pd.DataFrame(model_fit.resid)
        return RMSE, pred_active, residuals

    def ARIMA(self, method):
        method = method.upper()
        if method == 'AR':
            model = ARIMA(self.trainActiveCases, order=(2, 0, 0))
        elif method == 'MA':
            model = ARIMA(self.trainActiveCases, order=(0, 0, 2))
        elif method == 'ARIMA':
            model = ARIMA(self.trainActiveCases, order=(1, 1, 1))
        RMSE,pred_active,residuals = self.__ARIMA(model)
        print(color.BOLD + "\t{state}: Root Mean Square Error for {method} Model: {RMSE}".format(state=self.state, method=method, RMSE=str(RMSE)) + color.END)
        # print(residuals.describe())
        self.plotARIMA(pred_active,residuals,method)
        return RMSE, pred_active

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

    def plotARIMA(self,pred_active,residuals,title):
        # Plotting
        f, ax = plt.subplots(1,1 , figsize=(12,10))
        plt.plot(self.train_active, marker='o',color='blue',label ="Train Data Set")
        plt.plot(self.valid_active, marker='o',color='green',label ="Valid Data Set")
        plt.plot(pred_active, marker='o',color='red',label ="Predicted " + title)
        plt.legend()
        plt.xlabel("Date Time")
        plt.ylabel('Active Cases')
        plt.title("Active Cases {title} Model Forecasting for state {state}".format(state=self.state,title=title))
        plt.savefig(self.arima_path + self.last_date + '_{state}_{title}_Model.png'.format(state=self.state,title=title))
        # plot residual errors
        residuals.plot()
        resError = self.arima_path + '\\resError\\'
        plt.savefig(resError + self.last_date + '_{state}_{title}_Model_residual_error.png'.format(state=self.state,title=title))
        residuals.plot(kind='kde')
        plt.savefig(resError + self.last_date + '_{state}_{title}_Model_residual_error_kde.png'.format(state=self.state,title=title))
