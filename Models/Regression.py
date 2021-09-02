import numpy as np
import pandas as pd
import warnings as wn
import matplotlib.pyplot as plt

from Models import DATA_PATH, RESULTS_PATH

from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from tslearn.svm import TimeSeriesSVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class Regression:
    def __init__(self, state):
        self.state = state
        self.regression_path = RESULTS_PATH + '\\regression\\'

        self.df_per_State_features = pd.read_csv(DATA_PATH + state +'.csv')
        self.df_per_State_features = self.df_per_State_features.fillna(0)
        self.df_per_State_features["Active Cases"].replace({0:1}, inplace=True)

        data = self.df_per_State_features['Active Cases'].astype('double').values
        daterange = self.df_per_State_features['Date'].values
        date_index = pd.date_range(start=daterange[0], end=daterange[-1], freq='D')
        self.activecases = pd.Series(data, date_index)
        self.last_date = daterange[-1]

        df_per_State_sel_features = self.df_per_State_features.copy(deep=False)
        df_per_State_sel_features["Days Since"] = date_index - date_index[0]
        df_per_State_sel_features["Days Since"] = df_per_State_sel_features["Days Since"].dt.days
        df_per_State_sel_features = df_per_State_sel_features.iloc[:,[4,5, 7,8,9,10,11,12,13,14,15,16,23]]
        self.ml_all_f = df_per_State_sel_features.values
        self.train_ml_all_f = df_per_State_sel_features.iloc[:int(df_per_State_sel_features.shape[0]*0.70)].values
        self.valid_ml_all_f = df_per_State_sel_features.iloc[int(df_per_State_sel_features.shape[0]*0.70):].values

        self.df_per_State_features["Days Since"] = date_index - date_index[0]
        self.df_per_State_features["Days Since"] = self.df_per_State_features["Days Since"].dt.days
        self.totActiveCases = self.activecases.values.reshape(-1,1)
        self.trainActiveCases =self.totActiveCases[:int(self.df_per_State_features.shape[0]*0.70)]
        self.validActiveCases = self.totActiveCases[int(self.df_per_State_features.shape[0]*0.70):]

        self.train_ml = self.df_per_State_features.iloc[:int(self.df_per_State_features.shape[0]*0.70)]
        self.valid_ml = self.df_per_State_features.iloc[int(self.df_per_State_features.shape[0]*0.70):]
        self.train_dates = self.df_per_State_features['Date'].iloc[:int(df_per_State_sel_features.shape[0]*0.70)].values
        self.valid_dates = self.df_per_State_features['Date'].iloc[int(df_per_State_sel_features.shape[0]*0.70):].values

    def linearRegression(self):
        lin_reg=LinearRegression(normalize=True)
        lin_reg.fit(self.train_ml_all_f,self.trainActiveCases)
        prediction_valid_linreg=lin_reg.predict(self.valid_ml_all_f)
        RMSE = np.sqrt(mean_squared_error(self.validActiveCases,prediction_valid_linreg))
        print(self.state + ": Root Mean Square Error for Linear Regression: " + str(RMSE))
        prediction_linreg=lin_reg.predict(self.ml_all_f)
        self.plot(prediction_linreg,'Linear')
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
        self.plot(pred_poly,'Polynomial')
        return RMSE

    def lassoRegression(self):
        lasso_reg = Lasso(alpha=.8,normalize=True, max_iter=1e5)
        poly_reg = make_pipeline(PolynomialFeatures(3), lasso_reg)
        poly_reg.fit(self.train_ml_all_f,self.trainActiveCases)
        poly_pred = poly_reg.predict(self.valid_ml_all_f)
        RMSE = np.sqrt(mean_squared_error(self.validActiveCases,poly_pred))
        print(self.state + ": Root Mean Square Error for LASSO Regression: " + str(RMSE))
        pred_poly = poly_reg.predict(self.ml_all_f)
        self.plot(pred_poly,'LASSO')
        return RMSE

    def plot(self,prediction,model):
        plt.figure(figsize=(11,6))
        plt.plot(self.totActiveCases,label="Active Cases")
        plt.plot(self.df_per_State_features['Date'], prediction, linestyle='--',label="Predicted Active Cases using {model} Regression".format(model=model),color='black')
        plt.title("Active Cases {model} Regression Prediction".format(model=model))
        plt.xlabel('Time')
        plt.ylabel('Active Cases')
        plt.xticks(rotation=90)
        plt.legend()
        plt.savefig(self.regression_path + self.last_date + '_{state}_{model}_regression.png'.format(state=self.state,model=model.lower()))

    def tslean(self):
        reg = TimeSeriesSVR(kernel="gak", gamma="auto")
        all_colms = ['Cured/Discharged/Migrated', 'Death', 'Total Confirmed cases', 'LiteracyRate', 'PopulationDensity', 'ElderlyRate',
            'DistrictsEffected', 'NoRedZones', 'NoOrangeZones', 'NoGreenZones',
            'InternationalAirports', 'IntAirportPassenger', 'StateHospitals',
            'StateHospitalBeds', 'StateHospitalICUs', 'StateHospitalVentilators']
        colms = []
        no_colms = []
        for c in all_colms:
            if c in self.train_ml.columns:
                colms.append(c)
            else:
                no_colms.append(c)

        X_train = self.train_ml[colms]
        X_test =  self.valid_ml[colms]

        X = self.train_ml[[ 'Total Confirmed cases']]
        y =  self.valid_ml[['Total Confirmed cases']]

        tseries_reg = reg.fit(X_train, self.trainActiveCases)
        tseries_pred = tseries_reg.predict(X_test)
        # print(tseries_reg.support_vectors_time_series_(X_test))
        # print(tseries_pred)

        RMSE = np.sqrt(mean_squared_error(self.validActiveCases,tseries_pred))
        print("Root Mean Square Error for Tslearn Model: " + str(RMSE))

        index= pd.date_range(start=self.train_dates[0], periods=len(self.train_dates), freq='D')
        valid_index = pd.date_range(start=self.valid_dates[0], periods=len(self.valid_dates), freq='D')

        train_active =  pd.Series(self.train_ml['Active Cases'].values, index)
        valid_active =  pd.Series(self.valid_ml['Active Cases'].values, valid_index)
        pred_active =  pd.Series(tseries_pred, valid_index)

        f, ax = plt.subplots(1,1 , figsize=(12,10))
        plt.plot(train_active, marker='o',color='blue',label ="Train Data Set")
        plt.plot(valid_active, marker='o',color='green',label ="Valid Data Set")
        plt.plot(pred_active, marker='o',color='red',label ="Predicted Tslearn Active Cases")

        plt.legend()
        plt.xlabel("Date Time")
        plt.ylabel('Active Cases')
        plt.title("Multi Input Feature Active Cases Forecasting for state " + self.state)
        plt.savefig(self.regression_path + self.last_date + '_{state}_tslean.png'.format(state=self.state))
        return RMSE

