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



class Arima:
    def __init__(self, state):
        self.state = state
        self.model_scores = []
        self.arima_path = RESULTS_PATH + '\\arima\\'

        self.df_per_State_features = pd.read_csv(DATA_PATH + state +'.csv')
        self.df_per_State_features = self.df_per_State_features.fillna(0)
        self.df_per_State_features["Active Cases"].replace({0:1}, inplace=True)
        
        data = self.df_per_State_features['Active Cases'].astype('double').values
        daterange = self.df_per_State_features['Date'].values
        date_index = pd.date_range(start=daterange[0], end=daterange[-1], freq='D')
        self.activecases = pd.Series(data, date_index)
        self.last_date = daterange[-1]
        self.totActiveCases = self.activecases.values.reshape(-1,1)

        df_per_State_sel_features = self.df_per_State_features.copy(deep=False)
        df_per_State_sel_features["Days Since"] = date_index - date_index[0]
        df_per_State_sel_features["Days Since"] = df_per_State_sel_features["Days Since"].dt.days
        df_per_State_sel_features = df_per_State_sel_features.iloc[:,[4,5, 7,8,9,10,11,12,13,14,15,16,23]]

        self.train_ml = self.df_per_State_features.iloc[:int(self.df_per_State_features.shape[0]*0.70)]
        self.valid_ml = self.df_per_State_features.iloc[int(self.df_per_State_features.shape[0]*0.70):]
        self.trainActiveCases = self.totActiveCases[0:int(self.df_per_State_features.shape[0]*0.70)]
        self.validActiveCases = self.totActiveCases[int(self.df_per_State_features.shape[0]*0.70):]
        self.train_dates = self.df_per_State_features['Date'].iloc[:int(df_per_State_sel_features.shape[0]*0.70)].values
        self.valid_dates = self.df_per_State_features['Date'].iloc[int(df_per_State_sel_features.shape[0]*0.70):].values
    
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
        # self.model_scores.append(np.sqrt(mean_squared_error(self.validActiveCases,prediction_ar)))
        print("Root Mean Square Error for AR Model: " + str(np.sqrt(mean_squared_error(self.validActiveCases,prediction_ar))))
        # print(model_ar_fit.summary())
        # print(residuals.describe())

        index= pd.date_range(start=self.train_dates[0], periods=len(self.train_dates), freq='D')
        valid_index = pd.date_range(start=self.valid_dates[0], periods=len(self.valid_dates), freq='D')
        train_active =  pd.Series(self.train_ml['Active Cases'].values, index)
        valid_active =  pd.Series(self.valid_ml['Active Cases'].values, valid_index)
        pred_active =  pd.Series(prediction_ar, valid_index)
        
        # plot residual errors
        residuals = pd.DataFrame(model_ar_fit.resid)
        residuals.plot()
        plt.savefig(self.arima_path + self.last_date + '_{state}_AR_Model_residual_error{type}.png'.format(state=self.state,type=arima_type))
        residuals.plot(kind='kde')
        plt.savefig(self.arima_path + self.last_date + '_{state}_AR_Model_residual_error_kde{type}.png'.format(state=self.state,type=arima_type))
        # Plotting
        f, ax = plt.subplots(1,1 , figsize=(12,10))
        plt.plot(train_active, marker='o',color='blue',label ="Train Data Set")
        plt.plot(valid_active, marker='o',color='green',label ="Valid Data Set")
        plt.plot(pred_active, marker='o',color='red',label ="Predicted AR")
        plt.legend()
        plt.xlabel("Date Time")
        plt.ylabel('Active Cases')
        plt.title("Active Cases AR Model Forecasting for state " + self.state)
        plt.savefig(self.arima_path + self.last_date + '_{state}_AR_model{type}.png'.format(state=self.state,type=arima_type))
    
    def MA(self):
        if arima_type:
            wn.filterwarnings("ignore")
            model_ma = ARIMA(self.trainActiveCases, order=(0, 0, 2))
            wn.filterwarnings("default")
            model_ma_fit = model_ma.fit(disp=0)  
            prediction_ma=model_ma_fit.forecast(len(self.validActiveCases))[0]
        else:
            model_ma = ARIMA(self.trainActiveCases, order=(0, 0, 2))
            model_ma_fit = model_ma.fit()
            prediction_ma=model_ma_fit.forecast(len(self.validActiveCases))
        # self.model_scores.append(np.sqrt(mean_squared_error(self.validActiveCases,prediction_ma)))
        print("Root Mean Square Error for MA Model: " + str(np.sqrt(mean_squared_error(self.validActiveCases,prediction_ma))))
        # print(model_ma_fit.summary())
        # print(residuals.describe())

        index= pd.date_range(start=self.train_dates[0], periods=len(self.train_dates), freq='D')
        valid_index = pd.date_range(start=self.valid_dates[0], periods=len(self.valid_dates), freq='D')
        train_active =  pd.Series(self.train_ml['Active Cases'].values, index)
        valid_active =  pd.Series(self.valid_ml['Active Cases'].values, valid_index)
        pred_active =  pd.Series(prediction_ma, valid_index)
        
        # plot residual errors
        residuals = pd.DataFrame(model_ma_fit.resid)
        residuals.plot()
        plt.savefig(self.arima_path + self.last_date + '_{state}_MA_Model_residual_error{type}.png'.format(state=self.state,type=arima_type))
        residuals.plot(kind='kde')
        plt.savefig(self.arima_path + self.last_date + '_{state}_MA_Model_residual_error_kde{type}.png'.format(state=self.state,type=arima_type))
        # Plotting
        f, ax = plt.subplots(1,1 , figsize=(12,10))
        plt.plot(train_active, marker='o',color='blue',label ="Train Data Set")
        plt.plot(valid_active, marker='o',color='green',label ="Valid Data Set")
        plt.plot(pred_active, marker='o',color='red',label ="Predicted MA")
        plt.legend()
        plt.xlabel("Date Time")
        plt.ylabel('Active Cases')
        plt.title("Active Cases MA Model Forecasting for state " + self.state)
        plt.savefig(self.arima_path + self.last_date + '_{state}_MA_Model{type}.png'.format(state=self.state,type=arima_type))
    
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
        # self.model_scores.append(np.sqrt(mean_squared_error(self.validActiveCases,prediction_arima)))
        print("Root Mean Square Error for MA Model: " + str(np.sqrt(mean_squared_error(self.validActiveCases,prediction_arima))))
        # print(model_arima_fit.summary())
        # print(residuals.describe())

        # self.valid_ml["ARIMA Model Prediction"] = list(np.exp(prediction_arima))

        index = pd.date_range(start=self.train_dates[0], periods=len(self.train_dates), freq='D')
        valid_index = pd.date_range(start=self.valid_dates[0], periods=len(self.valid_dates), freq='D')
        train_active = pd.Series(self.train_ml['Active Cases'].values, index)
        valid_active = pd.Series(self.valid_ml['Active Cases'].values, valid_index)
        pred_active = pd.Series(prediction_arima, valid_index)

        # plot residual errors
        residuals = pd.DataFrame(model_arima_fit.resid)
        residuals.plot()
        plt.savefig(self.arima_path + self.last_date + '_{state}_ARIMA_Model_residual_error{type}.png'.format(state=self.state,type=arima_type))
        residuals.plot(kind='kde')
        plt.savefig(self.arima_path + self.last_date + '_{state}_ARIMA_Model_residual_error_kde{type}.png'.format(state=self.state,type=arima_type))
        # Plotting
        f, ax = plt.subplots(1,1 , figsize=(12,10))
        plt.plot(train_active, marker='o',color='blue',label ="Train Data Set")
        plt.plot(valid_active, marker='o',color='green',label ="Valid Data Set")
        plt.plot(pred_active, marker='o',color='red',label ="Predicted ARIMA")
        plt.legend()
        plt.xlabel("Date Time")
        plt.ylabel('Active Cases')
        plt.title("Active Cases ARIMA Model Forecasting for state " + self.state)
        plt.savefig(self.arima_path + self.last_date + '_{state}_ARIMA_Model{type}.png'.format(state=self.state,type=arima_type))

    def SARIMAX(self):
        model_sarima = sm.tsa.statespace.SARIMAX(self.trainActiveCases, order=(1, 1, 1), seasonal_order=(0,0,0,12), enforce_stationarity=False, enforce_invertibility=False)
        model_sarima_fit = model_sarima.fit(disp=0)  
        prediction_sarima=model_sarima_fit.forecast(np.shape(self.validActiveCases)[0])
        # self.model_scores.append(np.sqrt(mean_squared_error(self.validActiveCases,prediction_sarima)))
        print("Root Mean Square Error for SARIMA Model: " + str(np.sqrt(mean_squared_error(self.validActiveCases,prediction_sarima))))
        # print(model_sarima_fit.summary())
        # print(residuals.describe())

        # self.valid_ml["SARIMA Model Prediction"] = list(np.exp(prediction_sarima))
        # print(np.shape(self.validActiveCases))
        # print(np.shape(prediction_sarima[1:]))
        # print(np.shape(self.validActiveCases))

        index= pd.date_range(start=self.train_dates[0], periods=len(self.train_dates), freq='D')
        valid_index = pd.date_range(start=self.valid_dates[0], periods=len(self.valid_dates), freq='D')
        train_active =  pd.Series(self.train_ml['Active Cases'].values, index)
        valid_active =  pd.Series(self.valid_ml['Active Cases'].values, valid_index)
        pred_active =  pd.Series(prediction_sarima, valid_index)

        # plot residual errors
        residuals = pd.DataFrame(model_sarima_fit.resid)
        residuals.plot()
        plt.savefig(self.arima_path + self.last_date + '_{state}_SARIMAX_Model_residual_error.png'.format(state=self.state))
        residuals.plot(kind='kde')
        plt.savefig(self.arima_path + self.last_date + '_{state}_SARIMAX_Model_residual_error_kde.png'.format(state=self.state))
        # Plotting
        f, ax = plt.subplots(1,1 , figsize=(12,10))
        plt.plot(train_active, marker='o',color='blue',label ="Train Data Set")
        plt.plot(valid_active, marker='o',color='green',label ="Valid Data Set")
        plt.plot(pred_active, marker='o',color='red',label ="Predicted ARIMA")
        plt.legend()
        plt.xlabel("Date Time")
        plt.ylabel('Active Cases')
        plt.title("Active Cases SARIMAX Model Forecasting for state " + self.state)
        plt.savefig(self.arima_path + self.last_date + '_{state}_SARIMAX_Model.png'.format(state=self.state))



    def AUTO(self):
        pass
