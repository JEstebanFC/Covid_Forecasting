import numpy as np
import pandas as pd
import warnings as wn
import matplotlib.pyplot as plt

from Models import DATA_PATH, RESULTS_PATH

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

class Smoothing:
    def __init__(self, state):
        self.state = state
        self.smoothing_path = RESULTS_PATH + '\\smoothing\\'

        self.df_per_State_features = pd.read_csv(DATA_PATH + state +'.csv')
        self.df_per_State_features = self.df_per_State_features.fillna(0)
        self.df_per_State_features["Active Cases"].replace({0:1}, inplace=True)

        data = self.df_per_State_features['Active Cases'].astype('double').values
        daterange = self.df_per_State_features['Date'].values
        date_index = pd.date_range(start=daterange[0], end=daterange[-1], freq='D')
        self.activecases = pd.Series(data, date_index)
        self.no_Dates = len(daterange)
        self.last_date = daterange[-1]

        self.df_per_State_features["Days Since"] = date_index - date_index[0]
        self.df_per_State_features["Days Since"] = self.df_per_State_features["Days Since"].dt.days

    def exponentialSmoothing(self):
        len_forecast = np.shape(self.df_per_State_features['Active Cases'])[0]

        wn.filterwarnings("ignore")
        fit1 = SimpleExpSmoothing(self.df_per_State_features['Active Cases']).fit(smoothing_level=0.2,optimized=False)
        fcast1 = fit1.forecast(len_forecast).rename(r'$\alpha=0.2$')
        fit2 = SimpleExpSmoothing(self.df_per_State_features['Active Cases']).fit(smoothing_level=0.6,optimized=False)
        fcast2 = fit2.forecast(len_forecast).rename(r'$\alpha=0.6$')
        fit3 = SimpleExpSmoothing(self.df_per_State_features['Active Cases']).fit()
        fcast3 = fit3.forecast(len_forecast).rename(r'$\alpha=%s$'%fit3.model.params['smoothing_level'])
        fit4 = ExponentialSmoothing(self.df_per_State_features['Active Cases']).fit(damping_trend=.2, smoothing_level=.1)
        fcast4 = fit3.forecast(len_forecast).rename(r'$\alpha=%s$'%fit4.model.params['smoothing_level'])
        wn.filterwarnings("default")

        dateStart = self.df_per_State_features['Date'].values[0]
        index= pd.date_range(start=dateStart, periods=self.no_Dates, freq='D')
        pred_index= pd.date_range(start='2020-04-30', periods=len_forecast, freq='D')

        pred_fcast1 =  pd.Series(fcast1.values, pred_index)
        pred_fitval_fcast1 =  pd.Series(fit1.fittedvalues.values, index)

        pred_fcast2 =  pd.Series(fcast2.values, pred_index)
        pred_fitval_fcast2 =  pd.Series(fit2.fittedvalues.values, index)

        pred_fcast3 =  pd.Series(fcast3.values, pred_index)
        pred_fitval_fcast3 =  pd.Series(fit3.fittedvalues.values, index)

        pred_fcast4 =  pd.Series(fcast4.values, pred_index)
        pred_fitval_fcast4 =  pd.Series(fit4.fittedvalues.values, index)

        f, ax = plt.subplots(1,1 , figsize=(12,10))
        plt.plot(pred_fcast1, marker='o',color='blue',label ="Predicted Data Set " + r'$\alpha=0.2$')
        plt.plot(pred_fitval_fcast1, marker='^',color='blue',label ="Trained Data Set")

        plt.plot(self.activecases, marker='o', color='black', label ="Trained Active Cases Data Set")

        plt.plot(pred_fcast2, marker='o',color='red',label ="Predicted Data Set " + r'$\alpha=0.6$')
        plt.plot(pred_fitval_fcast2, marker='^',color='red',label ="Trained Data Set")

        plt.plot(pred_fcast3, marker='o',color='green',label ="Predicted Data Set " + r'$\alpha=%s$'%fit3.model.params['smoothing_level'])
        plt.plot(pred_fitval_fcast3, marker='^',color='green',label ="Trained Data Set")

        plt.plot(pred_fcast4, marker='o',color='magenta',label ="Predicted Data Set " r'$\alpha=%s$'%fit4.model.params['smoothing_level'])
        plt.plot(pred_fitval_fcast4, marker='^',color='magenta',label ="Trained Data Set")

        plt.legend()
        ax.set_title('Active case Flattening curve for ' + self.state)
        plt.savefig(self.smoothing_path + self.last_date + '_{state}_exponential_smoothing.png'.format(state=self.state))
    
    def holtsWinter(self):
        len_forecast = np.shape(self.df_per_State_features['Active Cases'])[0]
        wn.filterwarnings("ignore")
        fit1 = Holt(self.df_per_State_features['Active Cases']).fit(smoothing_level=1.1, smoothing_trend=1.6, optimized=False)
        fcast1 = fit1.forecast(len_forecast).rename("Holt's linear trend")
        fit2 = Holt(self.df_per_State_features['Active Cases'], exponential=True).fit(smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
        fcast2 = fit2.forecast(len_forecast).rename("Exponential trend")
        fit3 = Holt(self.df_per_State_features['Active Cases'], damped_trend=True).fit(smoothing_level=0.8, smoothing_trend=0.2)
        fcast3 = fit3.forecast(len_forecast).rename("Additive damped trend")
        wn.filterwarnings("default")

        dateStart = self.df_per_State_features['Date'].values[0]
        index= pd.date_range(start=dateStart, periods=self.no_Dates, freq='D')
        pred_index= pd.date_range(start='2020-04-30', periods=len_forecast, freq='D')

        pred_fcast1 =  pd.Series(fcast1.values, pred_index)
        pred_fitval_fcast1 =  pd.Series(fit1.fittedvalues.values, index)

        pred_fcast2 =  pd.Series(fcast2.values, pred_index)
        pred_fitval_fcast2 =  pd.Series(fit2.fittedvalues.values, index)

        pred_fcast3 =  pd.Series(fcast3.values, pred_index)
        pred_fitval_fcast3 =  pd.Series(fit3.fittedvalues.values, index)

        f, ax = plt.subplots(1,1 , figsize=(12,10))
        plt.plot(pred_fcast1, marker='o',color='blue',label ="Predicted Holt\'s Linear Trend")
        plt.plot(pred_fitval_fcast1, marker='^',color='blue',label ="Trained Holt\'s Linear Trend")

        plt.plot(self.activecases, marker='o', color='black', label ="Trained Active Cases Data Set")

        plt.plot(pred_fcast2, marker='o',color='red',label ="Predicted Holt\'s Exponential Trend")
        plt.plot(pred_fitval_fcast2, marker='^',color='red',label = "Trained Holt\'s Exponential Trend")

        plt.plot(pred_fcast3, marker='o',color='green',label ="Predicted Trained Holt\'s Additive Damped Trend")
        plt.plot(pred_fitval_fcast3, marker='^',color='green',label ="Trained Holt\'s Additive Damped Trend")

        plt.legend()
        ax.set_title('Holt\'s Winter Time Series Active Cases Prediction to Flatten the curve for ' + self.state)
        plt.savefig(self.smoothing_path + self.last_date + '_{state}_holts_winter.png'.format(state=self.state))

        wn.filterwarnings("ignore")
        fit1 = SimpleExpSmoothing(self.activecases).fit()
        fit2 = Holt(self.activecases).fit()
        fit3 = Holt(self.activecases,exponential=True).fit()
        fit4 = Holt(self.activecases,damped_trend=True).fit(damping_trend=0.98)
        fit5 = Holt(self.activecases,exponential=True,damped_trend=True).fit()
        wn.filterwarnings("default")
        
        params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'initial_level', 'initial_trend']
        results=pd.DataFrame(index=["\u03B1","\u03B2","\u03C6","l\u2080","b\u2080","SSE"] ,columns=['SES', "Holt's","Exponential", "Additive", "Multiplicative"])
        results["SES"] =            [fit1.params[p] for p in params] + [fit1.sse]
        results["Holt's"] =         [fit2.params[p] for p in params] + [fit2.sse]
        results["Exponential"] =    [fit3.params[p] for p in params] + [fit3.sse]
        results["Additive"] =       [fit4.params[p] for p in params] + [fit4.sse]
        results["Multiplicative"] = [fit5.params[p] for p in params] + [fit5.sse]
        print(results.round(4))

        title_sub = ['Holts', 'Holts Exponential', 'Holts Damped', 'Holts Exponential and Damped']
        count = 0
        for fit in [fit2, fit3, fit4, fit5]:
            wn.filterwarnings("ignore")
            ax = pd.DataFrame(np.c_[fit.level,fit.slope]).rename(columns={0:'level',1:'slope'}).plot(subplots=True)
            wn.filterwarnings("default")
            title = title_sub[count] + ' Level and Slope Curve for ' +  self.state
            ax[0].set_title(title)
            plt.savefig(self.smoothing_path + self.last_date + '_{state}_{sub}_level_slope.png'.format(state=self.state,sub=title_sub[count].replace(' ','_')))
            count = count + 1

