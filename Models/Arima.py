import numpy as np
import pandas as pd
import warnings as wn
import statsmodels.api as sm
import matplotlib.pyplot as plt

from Models import DATA_PATH, RESULTS_PATH

from sklearn.metrics import mean_squared_error


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

class Arima:
    def __init__(self, state):
        self.state = state
        self.arima_path = RESULTS_PATH + '\\arima\\'

        self.df_per_State_features = pd.read_csv(DATA_PATH + state +'.csv')
        self.df_per_State_features = self.df_per_State_features.fillna(0)
        self.df_per_State_features["Active Cases"].replace({0:1}, inplace=True)
        
        data = self.df_per_State_features['Active Cases'].astype('double').values
        daterange = self.df_per_State_features['Date'].values
        date_index = pd.date_range(start=daterange[0], end=daterange[-1], freq='D')
        
        self.last_date = daterange[-1]

        activecases = pd.Series(data, date_index)
        totActiveCases = activecases.values.reshape(-1,1)
        df_per_State_sel_features = self.df_per_State_features.copy(deep=False)
        df_per_State_sel_features["Days Since"] = date_index - date_index[0]
        df_per_State_sel_features["Days Since"] = df_per_State_sel_features["Days Since"].dt.days
        df_per_State_sel_features = df_per_State_sel_features.iloc[:,[4,5, 7,8,9,10,11,12,13,14,15,16,23]]

        # 70% training data, 30 for validation
        train_ml = self.df_per_State_features.iloc[:int(self.df_per_State_features.shape[0]*0.70)]
        valid_ml = self.df_per_State_features.iloc[int(self.df_per_State_features.shape[0]*0.70):]
        train_dates = self.df_per_State_features['Date'].iloc[:int(df_per_State_sel_features.shape[0]*0.70)].values
        valid_dates = self.df_per_State_features['Date'].iloc[int(df_per_State_sel_features.shape[0]*0.70):].values
        self.trainActiveCases = totActiveCases[0:int(self.df_per_State_features.shape[0]*0.70)]
        self.validActiveCases = totActiveCases[int(self.df_per_State_features.shape[0]*0.70):]

        self.train_index = pd.date_range(start=train_dates[0], periods=len(train_dates), freq='D')
        self.valid_index = pd.date_range(start=valid_dates[0], periods=len(valid_dates), freq='D')
        self.train_active = pd.Series(train_ml['Active Cases'].values, self.train_index)
        self.valid_active = pd.Series(valid_ml['Active Cases'].values, self.valid_index)
    
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
        self.plot(pred_active,residuals,'AR' + str(arima_type))
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
        self.plot(pred_active,residuals,'MA' + str(arima_type))
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
        # print(color.BOLD + "\t{state}: Root Mean Square Error for {method} Model: {RMSE}".format(state=self.state, method=method, RMSE=str(RMSE)) + color.END)
        pred_active = pd.Series(prediction_arima, self.valid_index)
        residuals = pd.DataFrame(model_arima_fit.resid)
        self.plot(pred_active,residuals,'ARIMA' + str(arima_type))
        # print(residuals.describe())
        # valid_ml["ARIMA Model Prediction"] = list(np.exp(prediction_arima))
        return pred_active,model_arima_fit,RMSE

    def SARIMAX(self):
        model_sarima = sm.tsa.statespace.SARIMAX(self.trainActiveCases, order=(1, 1, 1), seasonal_order=(0,0,0,12), enforce_stationarity=False, enforce_invertibility=False)
        model_sarima_fit = model_sarima.fit(disp=0)  
        prediction_sarima = model_sarima_fit.forecast(np.shape(self.validActiveCases)[0])
        RMSE = np.sqrt(mean_squared_error(self.validActiveCases,prediction_sarima))
        print(color.BOLD + "\tRoot Mean Square Error for SARIMA Model: " + str(RMSE) + color.END)
        pred_active = pd.Series(prediction_sarima, self.valid_index)
        residuals = pd.DataFrame(model_sarima_fit.resid)
        self.plot(pred_active,residuals,'SARIMAX')
        # print(residuals.describe())
        # valid_ml["SARIMA Model Prediction"] = list(np.exp(prediction_sarima))
        return pred_active,model_sarima_fit,RMSE

    def plot(self,pred_active,residuals,title):
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
        plt.savefig(self.arima_path + self.last_date + '_{state}_{title}_Model_residual_error.png'.format(state=self.state,title=title))
        residuals.plot(kind='kde')
        plt.savefig(self.arima_path + self.last_date + '_{state}_{title}_Model_residual_error_kde.png'.format(state=self.state,title=title))

    def AUTO(self):
        pass
