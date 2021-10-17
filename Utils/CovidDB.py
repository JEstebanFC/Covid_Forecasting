import os
import pandas as pd
import matplotlib.pyplot as plt

from Models import DATA_PATH, RESULTS_PATH

from datetime import datetime, timedelta

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

class CovidDB:
    def __init__(self):
        url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
        url_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
        url_recovered = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
        url_vaccine_doses = 'https://raw.githubusercontent.com/govex/COVID-19/master/data_tables/vaccine_data/global_data/time_series_covid19_vaccine_doses_admin_global.csv'
        url_vaccine_detail = 'https://raw.githubusercontent.com/govex/COVID-19/master/data_tables/vaccine_data/global_data/time_series_covid19_vaccine_global.csv'
        
        self.confirmed = pd.read_csv(url_confirmed, index_col=1)
        self.deaths = pd.read_csv(url_deaths, index_col=1)
        self.recovered = pd.read_csv(url_recovered, index_col=1)
        self.date_format = '%m/%d/%y'
        self.confirmed = self.confirmed.drop(columns=['Lat', 'Long'])
        
    def countryCases(self, country, date):
        '''This function return the new cases in the specific date and country'''
        index = 1
        today = datetime.strptime(date, self.date_format)
        yesterday = today - timedelta(days=1)
        lastDate = yesterday.strftime(self.date_format)
        if lastDate[0] == '0':
            lastDate = lastDate[1:]
        
        # confirmed_cases = self.confirmed.loc[self.confirmed[self.confirmed.columns[index]] == country]
        confirmed_cases = self.confirmed.loc[country]
        death_cases = self.deaths.loc[country]
        recover_cases = self.recovered.loc[country]

        daily_confirmed_cases = confirmed_cases[date] - confirmed_cases[lastDate]
        daily_death_cases = death_cases[date] - death_cases[lastDate]
        daily_recover_cases = recover_cases[date] - recover_cases[lastDate]

        daily_cases = [confirmed_cases.iloc[:,0]]
        daily_cases.append(pd.Series(daily_confirmed_cases,name='Confirmed'))
        daily_cases.append(pd.Series(daily_death_cases,name='Death'))
        daily_cases.append(pd.Series(daily_recover_cases,name='Recovered'))
        
        result = pd.concat(daily_cases, axis=1)
        print(color.BOLD  + 'Cases on {date}:'.format(country=country,lastDate=lastDate,date=date) + color.END)
        print(result.to_string(),'\n')
        return daily_confirmed_cases,daily_death_cases,daily_recover_cases

    def accumulativeCases(self, countries):
        '''Return the accumulative cases'''
        if type(countries) != list:
            countries = [countries]
        accumulativeCases = self.confirmed.loc[countries]
        return accumulativeCases

    def dailyCases(self, countries):
        '''
            Return the daily cases
            Example ['New Zealand','Autralia','India']: 
            Daily total cases in Autralia:
                dailyCases.loc['Australia'].sum()
            Daily cases in Australia-Victoria:
                dailyCases.loc['Australia'].loc['Victoria']
            Daily cases in New Zealand (no cook island):
                dailyCases.loc['New Zealand'].loc['']
            Daily cases in India:
                dailyCases.loc['India'].loc['']
        '''
        accumulativeCases = self.accumulativeCases(countries)
        ac = accumulativeCases.reset_index()
        ac['Province/State'] = ac['Province/State'].fillna('')
        ac = ac.set_index(['Country/Region','Province/State'])
        dailyCases = ac.diff(axis=1)
        dailyCases.columns = pd.to_datetime(dailyCases.columns)
        return dailyCases
        #data = dailyCases.loc['New Zealand'].loc['']
        #data.plot().get_figure().savefig(results_path + 'test1.png')

    def plotDailyCases(self, countries):
        if type(countries) != list:
            countries = [countries]
        for country in countries:
            dailyCases = self.dailyCases(country)
            try:
                data = dailyCases.loc[country].loc['']
            except:
                data = dailyCases.loc[country].sum()
            f, ax = plt.subplots(1,1, figsize=(12,10))
            plt.plot(data)
            ax.set_title('Active case History for ' + country)
            ax.set_ylabel("No of Active Covid-19 Cases")
            date = str(data.index[-1]).split()[0]
            results_path = self.createFolder(date)
            plt.savefig(results_path + date + '_{country}_active_cases.png'.format(country=country))

    def createFolder(self, date):
        results_path = '%sIT819\\active_cases\\%s\\' %(RESULTS_PATH, date)
        try:
            os.makedirs(results_path)
        except OSError:
            pass
        finally:
            # print('Results saved in: ', results_path)
            return results_path