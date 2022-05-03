import os
import logging
import numpy as np
from numpy.testing._private.utils import print_assert_equal
import pandas as pd
import warnings as wn
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

from datetime import datetime

from Utils.CovidDB import CovidDB

from Models import RESULTS_PATH

from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from keras.models import Sequential
from keras.layers import Dense, LSTM

from fbprophet import Prophet

class Models:
    def __init__(self, country):
        self.country = country
        self.covidDB = CovidDB()
        self.extra = '_'
        
    def selectData(self, initDay=None, lastDay=None, forecast=0, train_percent='0.75', plot=False):
        if 'w' in train_percent:
            train_weeks = int(train_percent.split('w')[0])
            train_percent = float(1.0)
        else:
            train_weeks = int(0)
            train_percent = float(train_percent)
        self.activecases, self.daysSince = self.getDailyCases(initDay=initDay, lastDay=lastDay, plot=plot)
        if self.activecases.empty:
            return pd.Series(dtype='object')
        lastDay = self.daysSince.values[-1]
        forecast_index = pd.date_range(start=self.daysSince.index[-1], periods=forecast+1, freq='D')[1:]
        self.forecastDays = pd.Series(range(lastDay+1, lastDay+1+forecast), forecast_index)
        if forecast > 0:
            self.extra += 'F' + str(forecast)

        L = self.activecases.shape[0]
        self.train_quantity = int(L*train_percent - train_weeks*7)
        self.train_ml = self.activecases[:self.train_quantity]
        self.valid_ml = self.activecases[self.train_quantity:]
        self.train_index = self.daysSince[:self.train_quantity]
        self.valid_index = self.daysSince[self.train_quantity:]
        self.train_active = self.train_ml.values.reshape(-1,1)
        self.valid_active = self.valid_ml.values.reshape(-1,1)
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
        dailyCases = self.covidDB.normalize(dailyCases)

        if not lastDay or lastDay not in dailyCases:        #Assuming there is information for every day inside the range
            lastDay = dailyCases.index[-1]
        lastDay = pd.to_datetime(lastDay)
        if initDay != None and 'w' in initDay:
            self.extra += str(initDay)
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

    def plot(self, xData, yData, lineStyle, legends, labels, fileName, title, plotLimit=True, **opts):
        if plotLimit:
            weekLimit = 30  #Week limit to show in plot
        else:
            weekLimit = 0
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111)
        plt.title(title)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.xticks(rotation=45)
        for xd,yd,ls,l in zip(xData, yData, lineStyle, legends):
            plt.plot(xd[-weekLimit*7:], yd[-weekLimit*7:], ls, label=l)
        if 'vertical' in opts:
            for ver in opts['vertical']:
                plt.axvline(x=ver[0], color='k', linestyle='--')
                ax.text(ver[0],ax.dataLim.max[-1],ver[1],size=14,horizontalalignment='right',color='green')
                ax.text(ver[0],ax.dataLim.max[-1],ver[2],size=14,horizontalalignment='left',color='green')
        plt.legend(loc=2)
        plt.savefig(self.plots_path + fileName)

    def plotResidualsARIMA(self, residuals, model):
        residuals.plot()
        plt.savefig(self.resPath + '{country}_{model}_residual_error.png'.format(country=self.country, model=model))
        residuals.plot(kind='kde')
        plt.savefig(self.resPath + '{country}_{model}_residual_error_kde.png'.format(country=self.country, model=model))


    #METRICS
    def __errors(self,validCases,prediction):
        errors = {}
        errors['R2'] = r2_score(validCases,prediction)
        errors['MAE'] = mae(validCases,prediction)
        errors['MAPE'] = mape(validCases,prediction)
        errors['RMSE'] = np.sqrt(mse(validCases,prediction))
        errors['NRMSE'] = np.sqrt(mse(validCases,prediction)/mse(validCases,[np.array([0])]*len(validCases)))
        errors['WSM'] =  self.WSM(errors)
        return errors
    
    def WSM(self,errors):
        weights = {}
        weights['R2'] = -0.5
        weights['MAPE'] = 0.25
        weights['NRMSE'] = 0.25
        wsm = 0.5
        for e in weights:
            wsm += errors[e] * weights[e]
        return wsm


    #REGRESSION
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
        vertical = [[self.valid_index.index[0],'Training   ','  Validation']]
        if not self.forecastDays.empty:
            vertical.append([self.forecastDays.index[0],'','  Forecast'])
        linestyle = ['-C0','-r']
        legends = ['Daily Cases',"Predicted Daily Cases: {model} Regression".format(model=method)]
        labels = ['Date Time','Daily Cases']
        fileName = '{country}_regression_{model}{extra}.png'.format(country=self.country, model=method.lower(),extra=self.extra)
        title = "Daily Cases {model} Regression Prediction for {country}".format(country=self.country,model=method)
        self.plot(xData, yData, linestyle, legends, labels, fileName, title, vertical=vertical)
        return errors,pred


    #ARIMA
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
        # model = ARIMA(history, order=order)
        # model_fit = model.fit()
        # if not self.forecastDays.empty:
        #     forecast = model_fit.forecast(len(self.forecastDays))
        forecast = pd.Series(forecast, self.forecastDays.index)
        pred = pd.Series(preds, self.valid_index)
        residuals = pd.DataFrame(model_fit.resid)
        return errors, pred, forecast, residuals

    def ARIMA(self):
        method = 'ARIMA'
        model = auto_arima(self.activecases, test='adf', suppress_warnings=True)
        order = model.order
        errors,pred,forecast,residuals = self.__ARIMA(order)
        # Plotting
        xData = [self.activecases.index, self.valid_index.index]
        yData = [self.activecases.values, pred]
        vertical = [[self.valid_index.index[0],'Training   ','  Validation']]
        linestyle = ['o-C0','o-r']
        legends = ['Daily Cases', 'Predicted '+ method + str(order).replace(' ','')]
        if len(forecast) != 0:
            xData.append(self.forecastDays.index)
            yData.append(forecast)
            linestyle.append('*-k')
            legends.append('{days} days of Forecast'.format(days=len(self.forecastDays)))
            vertical.append([self.forecastDays.index[0],'','  Forecast'])
            forecast.to_csv(self.csv_path + '{country}_{model}_forecast{extra}.csv'.format(country=self.country,model=method,extra=self.extra))
        labels = ['Date Time','Daily Cases']
        fileName = '{country}_{model}{extra}.png'.format(country=self.country, model=method + str(order),extra=self.extra)
        title = "Daily Cases {model} Model Forecasting for {country}".format(country=self.country,model=method)
        self.plot(xData, yData, linestyle, legends, labels, fileName, title, vertical=vertical)
        self.plotResidualsARIMA(residuals, method + str(order).replace(' ',''))
        pred.to_csv(self.csv_path + '{country}_{model}_validation{extra}.csv'.format(country=self.country,model=method,extra=self.extra))
        return errors,pred,forecast


    #Prophet
    def __prophet2(self):
        # Alternative algorithm, not working well
        history = pd.DataFrame({'ds':self.train_ml.index, 'y':self.train_ml.values})
        foreIndex = pd.to_datetime(self.valid_ml.index.to_list() + self.forecastDays.index.to_list())
        forecasting = pd.DataFrame({'ds':foreIndex})
        with suppress_stdout_stderr():
            model = Prophet()
            model.fit(history)
            prediction = model.predict(forecasting)
        prediction.set_index('ds',inplace=True)
        preds = prediction.yhat.loc[self.valid_ml.index[0]:self.valid_ml.index[-1]]
        forecast = prediction.yhat.loc[self.forecastDays.index[0]:self.forecastDays.index[-1]]
        errors = self.__errors(self.valid_active,preds.values.reshape(-1,1))
        return errors,preds,forecast

    def __prophet(self):
        preds = pd.DataFrame({'ds':[],'y':[]})
        history = pd.DataFrame({'ds': self.train_ml.index, 'y': self.train_ml.values})
        valid = pd.DataFrame({'ds':self.valid_ml.index, 'y':self.valid_ml.values})
        logging.getLogger('fbprophet').setLevel(logging.WARNING)
        for i in valid.index:
            val = valid.loc[i:i]
            with suppress_stdout_stderr():
                model = Prophet()
                model.fit(history)
                prediction = model.predict(val)
            preds = preds.append({'ds':prediction['ds'][0], 'y':prediction['yhat'][0]}, ignore_index=True)
            history = history.append({'ds': val['ds'][i], 'y': val['y'][i]}, ignore_index=True)
        errors = self.__errors(self.valid_active,preds['y'].values.reshape(-1,1))
        forecast = pd.DataFrame({'ds':self.forecastDays.index, 'y':None})
        for i in forecast.index:
            print(i)
            fore = forecast.loc[i:i]
            with suppress_stdout_stderr():
                model = Prophet()
                model.fit(history)
                prediction = model.predict(fore)
            forecast.y.loc[i:i] = prediction['yhat'][0]
            history = history.append({'ds': prediction['ds'][0], 'y': prediction['yhat'][0]}, ignore_index=True)
        preds.set_index('ds',inplace=True)
        forecast.set_index('ds',inplace=True)
        return errors,preds,forecast

    def prophet(self):
        errors,pred,forecast = self.__prophet()
        # Plotting
        method = 'Prophet'
        xData = [self.activecases.index, self.valid_index.index]
        yData = [self.activecases.values, pred]
        vertical = [[self.valid_index.index[0],'Training   ','  Validation']]
        linestyle = ['o-C0','o-r']
        legends = ['Daily Cases', 'Predicted '+ method]
        if len(forecast) != 0:
            xData.append(self.forecastDays.index)
            yData.append(forecast)
            linestyle.append('*-k')
            legends.append('{days} days of Forecast'.format(days=len(self.forecastDays)))
            vertical.append([self.forecastDays.index[0],'','  Forecast'])
            forecast.to_csv(self.csv_path + '{country}_{model}_forecast{extra}.csv'.format(country=self.country,model=method,extra=self.extra))
        labels = ['Date Time','Daily Cases']
        fileName = '{country}_{model}{extra}.png'.format(country=self.country, model=method,extra=self.extra)
        title = "Daily Cases {model} Model Forecasting for {country}".format(country=self.country,model=method)
        self.plot(xData, yData, linestyle, legends, labels, fileName, title, vertical=vertical)
        pred.to_csv(self.csv_path + '{country}_{model}_validation{extra}.csv'.format(country=self.country,model=method,extra=self.extra))
        return errors,pred,forecast


    #LSTM
    def timeseries_to_supervised(self, data, lag=1):
        df = pd.DataFrame(data)
        columns = [df.shift(i) for i in range(1, lag+1)]
        columns.append(df)
        df = pd.concat(columns, axis=1)
        df.fillna(0, inplace=True)
        # df.fillna(0, inplace=True)
        # df = df.drop(0)
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

    def invert_scale(self, scaler, X, value):
        new_row = [x for x in X] + [value]
        array = np.array(new_row)
        array = array.reshape(1, len(array))
        inverted = scaler.inverse_transform(array)
        return inverted[0, -1]

    def inverse_difference(self, history, yhat, interval=1):
        return yhat + history[-interval]

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
            if yhat < 0:
                yhat *= -1
            forecast.append(yhat)
            history = np.array([yhat])
        pred = pd.Series(predictions, self.valid_index)
        forecast = pd.Series(forecast, self.forecastDays.index)
        return errors, pred, forecast

    def LSTM(self):
        method = 'LSTM'
        diff_values = self.difference(self.activecases, 1)
        supervised = self.timeseries_to_supervised(diff_values, 1)
        supervised_values = supervised.values
        train = supervised_values[:self.train_quantity]
        test = supervised_values[self.train_quantity:]
        scaler, train_scaled, test_scaled = self.scale(train, test)
        lstm_model = self.fit_lstm(train_scaled, 1, 300, 4)
        # lstm_model = self.fit_lstm(train_scaled, 1, 3000, 4)
        train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
        self.temp = lstm_model.predict(train_reshaped, batch_size=1)
        errors,pred,forecast = self.__LSTM(lstm_model,test_scaled,scaler)

        xData = [self.activecases.index, self.valid_index.index]
        yData = [self.activecases.values, pred]
        vertical = [[self.valid_index.index[0],'Training   ','  Validation']]
        linestyle = ['o-C0','o-r']
        legends = ['Daily Cases', 'Predicted '+ method]
        if len(forecast) != 0:
            xData.append(self.forecastDays.index)
            yData.append(forecast)
            linestyle.append('*-k')
            legends.append('{days} days of Forecast'.format(days=len(self.forecastDays)))
            vertical.append([self.forecastDays.index[0],'','  Forecast'])
            forecast.to_csv(self.csv_path + '{country}_{model}_forecast{extra}.csv'.format(country=self.country,model=method,extra=self.extra))
        labels = ['Date Time','Daily Cases']
        fileName = '{country}_{model}{extra}.png'.format(country=self.country, model=method.lower(),extra=self.extra)
        title = "Daily Cases {model} Prediction for {country}".format(country=self.country,model=method)
        self.plot(xData, yData, linestyle, legends, labels, fileName, title, vertical=vertical)
        pred.to_csv(self.csv_path + '{country}_{model}_validation{extra}.csv'.format(country=self.country,model=method,extra=self.extra))
        # print(forecast.to_string())
        return errors,pred,forecast

#SuppresComments
class suppress_stdout_stderr(object):
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)