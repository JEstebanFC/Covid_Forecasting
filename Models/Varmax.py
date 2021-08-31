import numpy as np
import pandas as pd
import warnings as wn
import matplotlib.pyplot as plt

from Models import DATA_PATH, RESULTS_PATH

from statsmodels.tsa.statespace.varmax import VARMAX

class Varmax:
    def __init__(self, state):
        self.state = state
        self.model_scores = []
        self.varmax_path = RESULTS_PATH + '\\varmax\\'

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

        train_ml = self.df_per_State_features.iloc[:int(self.df_per_State_features.shape[0]*0.70)]
        valid_ml = self.df_per_State_features.iloc[int(self.df_per_State_features.shape[0]*0.70):]
        self.trainActiveCases = self.totActiveCases[0:int(self.df_per_State_features.shape[0]*0.70)]
        self.validActiveCases = self.totActiveCases[int(self.df_per_State_features.shape[0]*0.70):]
        self.train_dates = self.df_per_State_features['Date'].iloc[:int(df_per_State_sel_features.shape[0]*0.70)].values
        self.valid_dates = self.df_per_State_features['Date'].iloc[int(df_per_State_sel_features.shape[0]*0.70):].values
        self.df_per_State_features.index = self.df_per_State_features.Date
        self.train_var_ml = train_ml[['Cured/Discharged/Migrated','Death', 'Active Cases']]
        self.valid_var_ml = valid_ml[['Cured/Discharged/Migrated','Death', 'Active Cases', 'Total Confirmed cases']]
        # print(train_ml.columns)

    def AR(self):
        mod = VARMAX(endog=self.train_var_ml, order=(2,0), trend='n')#exog=exog
        res = mod.fit(maxiter=1000, disp=False)
        # print(res.summary())

        valid_index = pd.date_range(start=self.valid_dates[0], periods=61, freq='D')
        pred = res.impulse_responses(60, orthogonalized=True)

        a_vr_Series = pd.Series(pred['Active Cases'].values, index=valid_index)
        c_vr_Series = pd.Series(pred['Cured/Discharged/Migrated'].values, index=valid_index)
        d_vr_Series = pd.Series(pred['Death'].values, index=valid_index)

        f, ax = plt.subplots(1,1 , figsize=(12,10))
        plt.plot(a_vr_Series, marker='o',color='blue',label ="Aactive Cases")
        plt.plot(c_vr_Series, marker='o',color='green',label ="Cured/Discharged/Migrated")
        plt.plot(d_vr_Series, marker='o',color='red',label ="Death")

        plt.legend()
        plt.xlabel("Date Time")
        plt.ylabel('Total Nuner of Incident Cases')
        plt.title("VARMAX (Vector Autoregressive Model with Exogenous Variables) based AR Forecasting for state " + self.state)
        plt.savefig(self.varmax_path + self.last_date + '_{state}_AR_Varmax.png'.format(state=self.state))

    def VMA(self):
        mod = VARMAX(endog=self.train_var_ml, order=(0,2),  trend='n', error_cov_type='diagonal')
        res = mod.fit(maxiter=100, disp=False)
        # print(res.summary())

        valid_index = pd.date_range(start=self.valid_dates[0], periods=61, freq='D')
        pred = res.impulse_responses(60, orthogonalized=True)

        a_vr_Series = pd.Series(pred['Active Cases'].values, index=valid_index)
        c_vr_Series = pd.Series(pred['Cured/Discharged/Migrated'].values, index=valid_index)
        d_vr_Series = pd.Series(pred['Death'].values, index=valid_index)

        f, ax = plt.subplots(1,1 , figsize=(12,10))
        plt.plot(a_vr_Series, marker='o',color='blue',label ="Aactive Cases")
        plt.plot(c_vr_Series, marker='o',color='green',label ="Cured/Discharged/Migrated")
        plt.plot(d_vr_Series, marker='o',color='red',label ="Death")

        plt.legend()
        plt.xlabel("Date Time")
        plt.ylabel('Total Nuner of Incident Cases')
        plt.title("VARMAX (Vector Autoregressive Model with Exogenous Variables) based MA Forecasting for state " + self.state)
        plt.savefig(self.varmax_path + self.last_date + '_{state}_VMA_Varmax.png'.format(state=self.state))

    def ARIMA(self):
        mod = VARMAX(endog=self.train_var_ml, order=(2,2),  trend='n', error_cov_type='diagonal')
        res = mod.fit(maxiter=1000, disp=False)
        # print(res.summary())

        valid_index = pd.date_range(start=self.valid_dates[0], periods=61, freq='D')
        pred = res.impulse_responses(60, orthogonalized=True)

        a_vr_Series = pd.Series(pred['Active Cases'].values, index=valid_index)
        c_vr_Series = pd.Series(pred['Cured/Discharged/Migrated'].values, index=valid_index)
        d_vr_Series = pd.Series(pred['Death'].values, index=valid_index)

        f, ax = plt.subplots(1,1 , figsize=(12,10))
        plt.plot(a_vr_Series, marker='o',color='blue',label ="Aactive Cases")
        plt.plot(c_vr_Series, marker='o',color='green',label ="Cured/Discharged/Migrated")
        plt.plot(d_vr_Series, marker='o',color='red',label ="Death")

        plt.legend()
        plt.xlabel("Date Time")
        plt.ylabel('Total Nuner of Incident Cases')
        plt.title("VARMAX (Vector Autoregressive Model with Exogenous Variables) based ARIMA Forecasting for state " + self.state)
        plt.savefig(self.varmax_path + self.last_date + '_{state}_ARIMA_Varmax.png'.format(state=self.state))

