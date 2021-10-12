import numpy as np
import pandas as pd
import warnings as wn
import statsmodels.api as sm
import matplotlib.pyplot as plt

from Models import DATA_PATH, RESULTS_PATH

from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from tslearn.svm import TimeSeriesSVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

arima_type = 0
if arima_type:
    from statsmodels.tsa.arima_model import ARIMA
else:
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

    def linearRegression(self):
        lin_reg=LinearRegression(normalize=True)
        lin_reg.fit(self.train_ml_all_f,self.trainActiveCases)
        prediction_valid_linreg=lin_reg.predict(self.valid_ml_all_f)
        RMSE = np.sqrt(mean_squared_error(self.validActiveCases,prediction_valid_linreg))
        print(self.state + ": Root Mean Square Error for Linear Regression: " + str(RMSE))
        prediction_linreg=lin_reg.predict(self.ml_all_f)
        self.plotRegression(prediction_linreg,'Linear')
        return RMSE

    def polynomialRegression(self):
        poly_reg = PolynomialFeatures(degree = 7) 
        poly_reg = make_pipeline(PolynomialFeatures(3), Ridge())
        wn.filterwarnings("ignore")
        poly_reg.fit(self.train_ml_all_f,self.trainActiveCases)
        wn.filterwarnings("default")
        poly_pred = poly_reg.predict(self.valid_ml_all_f)
        RMSE = np.sqrt(mean_squared_error(self.validActiveCases,poly_pred))
        print(self.state + ": Root Mean Square Error for Polynomial Regression: " + str(RMSE))
        pred_poly = poly_reg.predict(self.ml_all_f)
        self.plotRegression(pred_poly,'Polynomial')
        return RMSE

    def lassoRegression(self):
        lasso_reg = Lasso(alpha=.8,normalize=True, max_iter=1e5)
        poly_reg = make_pipeline(PolynomialFeatures(3), lasso_reg)
        poly_reg.fit(self.train_ml_all_f,self.trainActiveCases)
        poly_pred = poly_reg.predict(self.valid_ml_all_f)
        RMSE = np.sqrt(mean_squared_error(self.validActiveCases,poly_pred))
        print(self.state + ": Root Mean Square Error for LASSO Regression: " + str(RMSE))
        pred_poly = poly_reg.predict(self.ml_all_f)
        self.plotRegression(pred_poly,'LASSO')
        return RMSE

    def AR(self):
        if arima_type:
            wn.filterwarnings("ignore")
            model_ar = ARIMA(self.trainActiveCases, order=(2, 0, 0))
            wn.filterwarnings("default")
            model_ar_fit = model_ar.fit(disp=0)
            prediction_ar = model_ar_fit.forecast(len(self.validActiveCases))[0]
        else:
            model_ar = ARIMA(self.trainActiveCases, order=(2, 0, 0))
            model_ar_fit = model_ar.fit()
            prediction_ar = model_ar_fit.forecast(len(self.validActiveCases))
        RMSE = np.sqrt(mean_squared_error(self.validActiveCases,prediction_ar))
        print(color.BOLD + "\tRoot Mean Square Error for AR Model: " + str(RMSE) + color.END)
        pred_active = pd.Series(prediction_ar, self.valid_index)
        residuals = pd.DataFrame(model_ar_fit.resid)
        self.plotARIMA(pred_active,residuals,'AR' + str(arima_type))
        # print(residuals.describe())
        return pred_active,model_ar_fit,RMSE
        
    def MA(self):
        if arima_type:
            wn.filterwarnings("ignore")
            model_ma = ARIMA(self.trainActiveCases, order=(0, 0, 2))
            wn.filterwarnings("default")
            model_ma_fit = model_ma.fit(disp=0)  
            prediction_ma = model_ma_fit.forecast(len(self.validActiveCases))[0]
        else:
            model_ma = ARIMA(self.trainActiveCases, order=(0, 0, 2))
            model_ma_fit = model_ma.fit()
            prediction_ma = model_ma_fit.forecast(len(self.validActiveCases))
        RMSE = np.sqrt(mean_squared_error(self.validActiveCases,prediction_ma))
        print(color.BOLD + "\tRoot Mean Square Error for MA Model: " + str(RMSE) + color.END)
        pred_active = pd.Series(prediction_ma, self.valid_index)
        residuals = pd.DataFrame(model_ma_fit.resid)
        self.plotARIMA(pred_active,residuals,'MA' + str(arima_type))
        # print(residuals.describe())
        return pred_active,model_ma_fit,RMSE

    def ARIMA(self):
        if arima_type:
            wn.filterwarnings("ignore")
            model_arima = ARIMA(self.trainActiveCases, order=(1, 1, 1))
            wn.filterwarnings("default")
            model_arima_fit = model_arima.fit(disp=0)  
            prediction_arima = model_arima_fit.forecast(len(self.validActiveCases))[0]
        else:
            model_arima = ARIMA(self.trainActiveCases, order=(1, 1, 1))
            model_arima_fit = model_arima.fit()
            prediction_arima = model_arima_fit.forecast(len(self.validActiveCases))
        RMSE = np.sqrt(mean_squared_error(self.validActiveCases,prediction_arima))
        print(color.BOLD + "\tRoot Mean Square Error for ARIMA Model: " + str(RMSE) + color.END)
        pred_active = pd.Series(prediction_arima, self.valid_index)
        residuals = pd.DataFrame(model_arima_fit.resid)
        self.plotARIMA(pred_active,residuals,'ARIMA' + str(arima_type))
        # print(residuals.describe())
        # self.valid_ml["ARIMA Model Prediction"] = list(np.exp(prediction_arima))
        return pred_active,model_arima_fit,RMSE

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

