import os
import numpy as np
from numpy.testing._private.utils import print_assert_equal
import pandas as pd
import datetime as dt
import warnings as wn
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

from Utils.CovidDB import CovidDB

from Models import DATA_PATH, RESULTS_PATH

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from keras.models import Sequential
from keras.layers import Dense, LSTM

class Models:
    def __init__(self, country):
        self.country = country
        self.covidDB = CovidDB()
        
    def selectData(self, initDay=None, lastDay=None, forecast=0, train_percent=0.7, plot=False):
        self.activecases,self.daysSince = self.getDailyCases(initDay=initDay, lastDay=lastDay, plot=plot)
        if self.activecases.empty:
            return pd.Series(dtype='object')
        lastDay = self.daysSince.values[-1]
        forecast_index = pd.date_range(start=self.daysSince.index[-1], periods=forecast+1, freq='D')[1:]
        self.forecastDays = pd.Series(range(lastDay+1, lastDay+1+forecast), forecast_index)

        train_quantity = int(self.activecases.shape[0]*train_percent)
        train_ml = self.activecases[:train_quantity]
        valid_ml = self.activecases[train_quantity:]
        self.train_index = self.daysSince[:train_quantity]
        self.valid_index = self.daysSince[train_quantity:]
        self.train_active = train_ml.values.reshape(-1,1)
        self.valid_active = valid_ml.values.reshape(-1,1)
        return pd.Series(True)

    def getDailyCases(self, initDay=None, lastDay=None, plot=False):
        country = self.country
        dailyCases = self.covidDB.dailyCases(country)
        if dailyCases.empty:
            return pd.Series(dtype='object'),pd.Series(dtype='object')
        try:
            dailyCases = dailyCases.loc[country].loc['']
        except:
            dailyCases = dailyCases.loc[country].sum()
        
        if not lastDay or lastDay not in dailyCases:        #Assuming there is information for every day inside the range
            lastDay = dailyCases.index[-1]
        lastDay = pd.to_datetime(lastDay)
        if initDay != None and 'w' in initDay:
            weeks = int(initDay.split('w')[0])
            initDay = lastDay - pd.DateOffset(7*weeks)
        if initDay == None or initDay not in dailyCases:    #Assuming there is information for every day inside the range
            initDay = dailyCases.index[0]
        initDay = pd.to_datetime(initDay)
        dailyCases = dailyCases[initDay:lastDay]
        casesFrame = dailyCases.to_frame('Daily Cases')
        casesFrame["Days Since"] = casesFrame.index - casesFrame.index[0]
        casesFrame["Days Since"] = casesFrame["Days Since"].dt.days
        daysSince = pd.Series(casesFrame['Days Since'].values, casesFrame.index)
        #Reports
        new_folder = '{lastDay}\\{firstDay}'.format(lastDay=str(lastDay.date()),firstDay=str(initDay.date()))
        self.plots_path = self.covidDB.createFolder(new_folder)
        self.csv_path = self.covidDB.createFolder(new_folder + '\\csv')
        self.resPath = self.covidDB.createFolder(new_folder + '\\residual')
        casesFrame.to_csv(self.csv_path + '{country}_daily_cases.csv'.format(country=country))
        #Plotting Daily Cases
        if plot:
            xData = [dailyCases.index]
            yData = [dailyCases.values]
            linestyle = ['-C0']
            legends = ['Daily cases']
            labels = ['Date Time','Daily Cases']
            fileName = '{country}_daily_cases.png'.format(country=country)
            title = 'Daily case for ' + country
            self.plot(xData, yData, linestyle, legends, labels, fileName, title)
        return dailyCases,daysSince

    def __errors(self,validCases,prediction):
        RMSE = np.sqrt(mse(validCases,prediction))
        MAE = mae(validCases,prediction)
        R2 = r2_score(validCases,prediction)
        return [RMSE,MAE,R2]
    
    def __regression(self, method, poly=3):
        X_values = self.train_index.values.reshape(-1,1)
        if method.lower() == 'linear':
            regression = LinearRegression(normalize=True)
        elif method.lower() == 'lasso':
            lasso_reg = Lasso(alpha=.8,normalize=True, max_iter=1e5)
            regression = make_pipeline(PolynomialFeatures(3), lasso_reg)
        elif 'polynomial' in method.lower():
            t = method.lower().split('polynomial')[-1]
            if t:
                poly = int(t)
            regression = make_pipeline(PolynomialFeatures(degree=poly), LinearRegression())
            # regression = make_pipeline(PolynomialFeatures(degree=poly), Ridge())
        regression.fit(X_values,self.train_active)
        prediction = regression.predict(self.valid_index.values.reshape(-1,1))
        errors = self.__errors(self.valid_active,prediction)
        pred = regression.predict(self.daysSince.append(self.forecastDays).values.reshape(-1,1))
        return errors,pred

    def regression(self, method):
        method = method.capitalize()
        errors,pred = self.__regression(method)
        # Plotting
        xData = [self.activecases.index,self.activecases.index.append(self.forecastDays.index)]
        yData = [self.activecases.values,pred]
        vertical = [self.valid_index.index[0]]
        if not self.forecastDays.empty:
            vertical.append(self.forecastDays.index[0])
        linestyle = ['-C0','-r']
        legends = ['Daily Cases',"Predicted Daily Cases: {model} Regression".format(model=method)]
        labels = ['Date Time','Daily Cases']
        fileName = '{country}_regression_{model}.png'.format(country=self.country, model=method.lower())
        title = "Daily Cases {model} Regression Prediction for {country}".format(country=self.country,model=method)
        self.plot(xData, yData, linestyle, legends, labels, fileName, title, vertical=vertical)
        return errors,pred

    def __ARIMA(self, order):
        preds = []
        history = [x for x in self.train_active]
        for val in self.valid_active:
            model = ARIMA(history, order=order)
            model_fit = model.fit()
            prediction = model_fit.forecast()
            preds.append(prediction[0])
            history.append(val)
            # history.append(np.array([prediction[0]]))
        errors = self.__errors(self.valid_active,preds)
        forecast = []
        for fore in self.forecastDays:
            model = ARIMA(history, order=order)
            model_fit = model.fit()
            prediction = model_fit.forecast()
            forecast.append(prediction[0])
            history.append(np.array([prediction[0]]))
        # if not self.forecastDays.empty:
        #     forecast = model_fit.forecast(len(self.forecastDays))
        forecast = pd.Series(forecast, self.forecastDays.index)
        pred = pd.Series(preds, self.valid_index)
        residuals = pd.DataFrame(model_fit.resid)
        return errors, pred, forecast, residuals

    def ARIMA(self, order):
        method = 'ARIMA'
        errors,pred,forecast,residuals = self.__ARIMA(order)
        # Plotting
        xData = [self.activecases.index, self.valid_index.index]
        yData = [self.activecases.values, pred]
        vertical = [self.valid_index.index[0]]
        linestyle = ['o-C0','o-r']
        legends = ['Daily Cases', 'Predicted '+ method + str(order).replace(' ','')]
        if len(forecast) != 0:
            xData.append(self.forecastDays.index)
            yData.append(forecast)
            linestyle.append('*-k')
            legends.append('{days} of Forecast'.format(days=len(self.forecastDays)))
            vertical.append(self.forecastDays.index[0])
        labels = ['Date Time','Daily Cases']
        fileName = '{country}_{model}.png'.format(country=self.country, model=method + str(order))
        title = "Daily Cases {model} Model Forecasting for {country}".format(country=self.country,model=method)
        self.plot(xData, yData, linestyle, legends, labels, fileName, title, vertical=vertical)
        self.plotResidualsARIMA(residuals, method)
        return errors,pred,forecast

    def plot(self, xData, yData, lineStyle, legends, labels, fileName, title, **opts):
        plt.figure(figsize=(12,10))
        plt.title(title)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.xticks(rotation=45)
        for xd,yd,ls,l in zip(xData, yData, lineStyle, legends):
            plt.plot(xd, yd, ls, label=l)
        if 'vertical' in opts:
            for ver in opts['vertical']:
                plt.axvline(x=ver, color='k', linestyle='--')
        plt.legend(loc=2)
        plt.savefig(self.plots_path + fileName)

    def plotResidualsARIMA(self, residuals, model):
        residuals.plot()
        plt.savefig(self.resPath + '{country}_{model}_residual_error.png'.format(country=self.country, model=model))
        residuals.plot(kind='kde')
        plt.savefig(self.resPath + '{country}_{model}_residual_error_kde.png'.format(country=self.country, model=model))

    #LSTM
    def timeseries_to_supervised(self, data, lag=1):
        df = pd.DataFrame(data)
        columns = [df.shift(i) for i in range(1, lag+1)]
        columns.append(df)
        df = pd.concat(columns, axis=1)
        df.fillna(0, inplace=True)
        return df

    def difference(self, dataset, interval=1):
        diff = [0]                              #Assume first difference is zero to match length with valid data
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return pd.Series(diff)

    def scale(self, train, test):
        # fit scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train)
        # transform train
        train = train.reshape(train.shape[0], train.shape[1])
        train_scaled = scaler.transform(train)
        # transform test
        test = test.reshape(test.shape[0], test.shape[1])
        test_scaled = scaler.transform(test)
        return scaler, train_scaled, test_scaled

    def fit_lstm(self, train, batch_size, nb_epoch, neurons):
        X, y = train[:, 0:-1], train[:, -1]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        for i in range(nb_epoch):
            model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
            model.reset_states()
        return model

    def invert_scale(self, scaler, X, value):
        new_row = [x for x in X] + [value]
        array = np.array(new_row)
        array = array.reshape(1, len(array))
        inverted = scaler.inverse_transform(array)
        return inverted[0, -1]

    def inverse_difference(self, history, yhat, interval=1):
	    return yhat + history[-interval]

    def forecast_lstm(self, model, batch_size, X):
        X = X.reshape(1, 1, len(X))
        yhat = model.predict(X, batch_size=batch_size)
        return yhat[0,0]

    def __LSTM(self, model, test_scaled, scaler):
        predictions = list()
        for i in range(len(test_scaled)):
            # make one-step forecast
            X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
            yhat = self.forecast_lstm(model, 1, X)
            # invert scaling
            yhat = self.invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = self.inverse_difference(self.activecases.values, yhat, len(test_scaled)+1-i)
            if yhat < 0:
                yhat *= -1
            # store forecast
            predictions.append(yhat)
        errors = self.__errors(self.valid_active,predictions)
        #Forecasting
        forecast = list()
        history = test_scaled[-1,-1:]
        for i in range(len(self.forecastDays)):
            X = history
            yhat = self.forecast_lstm(model, 1, X)
            yhat = self.invert_scale(scaler, X, yhat)
            yhat = self.inverse_difference(self.activecases.values, yhat, len(self.forecastDays)+1-i)
            forecast.append(yhat)
            history = np.array([yhat])

        return errors, predictions, forecast


    def LSTM(self, train_percent=0.7):
        method = 'LSTM'
        train_quantity = int(self.activecases.shape[0]*train_percent)
        diff_values = self.difference(self.activecases, 1)
        supervised = self.timeseries_to_supervised(diff_values, 1)
        supervised_values = supervised.values
        train = supervised_values[:train_quantity]
        test = supervised_values[train_quantity:]
        scaler, train_scaled, test_scaled = self.scale(train, test)
        lstm_model = self.fit_lstm(train_scaled, 1, 3000, 4)
        train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
        self.temp = lstm_model.predict(train_reshaped, batch_size=1)
        errors,predictions,forecast = self.__LSTM(lstm_model,test_scaled,scaler)
        

        xData = [self.activecases.index, self.valid_index.index]
        yData = [self.activecases.values, predictions]
        vertical = [self.valid_index.index[0]]
        linestyle = ['o-C0','o-r']
        legends = ['Daily Cases', 'Predicted '+ method]
        if len(forecast) != 0:
            xData.append(self.forecastDays.index)
            yData.append(forecast)
            linestyle.append('*-k')
            legends.append('{days} of Forecast'.format(days=len(self.forecastDays)))
            vertical.append(self.forecastDays.index[0])
        labels = ['Date Time','Daily Cases']
        fileName = '{country}_{model}.png'.format(country=self.country, model=method.lower())
        title = "Daily Cases {model} Prediction for {country}".format(country=self.country,model=method)
        self.plot(xData, yData, linestyle, legends, labels, fileName, title, vertical=vertical)
        return errors,predictions,forecast

