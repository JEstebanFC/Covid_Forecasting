import numpy as np
import pandas as pd
import warnings as wn
import statsmodels.api as sm
import matplotlib.pyplot as plt

from Models import DATA_PATH, RESULTS_PATH

from statsmodels.tsa.stattools import adfuller

class DickeyFuller:
    def __init__(self, state):
        self.state = state
        self.dickeyfuller_path = RESULTS_PATH + '\\dickeyfuller\\'

        self.df_per_State_features = pd.read_csv(DATA_PATH + state +'.csv')
        self.df_per_State_features = self.df_per_State_features.fillna(0)
        self.df_per_State_features["Active Cases"].replace({0:1}, inplace=True)

        data = self.df_per_State_features['Active Cases'].astype('double').values
        daterange = self.df_per_State_features['Date'].values
        date_index = pd.date_range(start=daterange[0], end=daterange[-1], freq='D')
        self.activecases = pd.Series(data, date_index)
        self.last_date = daterange[-1]

    def dickeyFuller(self):
        dftest = adfuller(self.activecases, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print("\nResults of Dickey-Fuller test for Original Time Series")
        print(dfoutput.round(4))

        log_series=np.log(self.activecases)
        dftest = adfuller((log_series.diff().diff()).dropna(), autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print("\nResults of Dickey-Fuller test for Log Transformed Time-Series")
        print(dfoutput.round(4))
        stationary=(log_series.diff().diff()).dropna()

        fig, (ax1,ax2,ax3) = plt.subplots(3, 1,figsize=(15,7))
        results=sm.tsa.seasonal_decompose(stationary)
        ax1.plot(results.trend)
        ax2.plot(results.seasonal)
        ax3.plot(results.resid)
        title='Dickey-Fuller test for ' + self.state
        fig.suptitle(title)
        plt.savefig(self.dickeyfuller_path + self.last_date + '_{state}_dickey_fuller.png'.format(state=self.state))

        plt.figure(figsize=(10, 5))
        title='Autocorrelation of active case for ' + self.state
        pd.plotting.autocorrelation_plot(self.activecases)
        plt.title(title)
        plt.savefig(self.dickeyfuller_path + self.last_date + '_{state}_active_cases_autocorr.png'.format(state=self.state))
