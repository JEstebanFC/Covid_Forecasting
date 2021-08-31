import numpy as np
import pandas as pd
import warnings as wn
import matplotlib.pyplot as plt

from Models import DATA_PATH, RESULTS_PATH

from fbprophet import Prophet
from sklearn.metrics import mean_squared_error

class Fbprophet:
    def __init__(self, state):
        self.state = state
        self.model_scores = []
        self.fbprophet_path = RESULTS_PATH + '\\fbprophet\\'

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
        self.df_per_State_features['Date'] = pd.to_datetime(self.df_per_State_features['Date'])

    def prophet(self):
        prophet_a=Prophet(interval_width=0.95,weekly_seasonality=True)
        prophet_active=pd.DataFrame(zip(list(self.df_per_State_features['Date']),list(self.df_per_State_features['Active Cases'])),columns=['ds','y'])

        prophet_a.fit(prophet_active)
        print(prophet_a)
        future_active = prophet_a.make_future_dataframe(periods=365)
        # future_active.tail()

        forecast_active = prophet_a.predict(future_active)
        forecast_active[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

        active_cases_forecast = prophet_a.predict(future_active)

        self.model_scores.append(np.sqrt(mean_squared_error(self.totActiveCases,forecast_active['yhat'].head(self.df_per_State_features.shape[0]))))
        print("Root Mean Squared Error for Prophet Model: " + str(np.sqrt(mean_squared_error(self.totActiveCases,forecast_active['yhat'].head(self.df_per_State_features.shape[0])))))

        fig1 = prophet_a.plot(active_cases_forecast)
        fig2 = prophet_a.plot_components(active_cases_forecast)

        fig1.savefig(self.fbprophet_path + self.last_date + '_{state}_Prophet.png'.format(state=self.state))
        fig2.savefig(self.fbprophet_path + self.last_date + '_{state}_Prophet_weekly.png'.format(state=self.state))

