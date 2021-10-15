import pandas as pd

from datetime import datetime, timedelta

from Models import DATA_PATH_NEW, RESULTS_PATH

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

    def newCasesReport(self, countries):
        '''Desgin to return a report with the new daily cases of a
        country, showing the states when this data is available
        '''
        start_index = 3
        statesColumn = self.confirmed.columns[0]
        confirmed_cases = self.confirmed.loc[countries]
        dailyCases = confirmed_cases[[statesColumn]].copy()
        L = len(confirmed_cases.columns)
        for i in range(start_index, L-1):
            dailyCases[confirmed_cases.columns[i+1]] = confirmed_cases[confirmed_cases.columns[i+1]] - confirmed_cases[confirmed_cases.columns[i]]
        
        return dailyCases
    
    def newCasesCountries(self, countries):
        '''
        Return the accumulative cases
        '''
        accumulativeCases = pd.DataFrame(columns=self.confirmed.columns)
        accumulativeCases = accumulativeCases.append(self.confirmed.loc[countries])
        return accumulativeCases

    def newDailyCasesCountries(self, countries):
        '''
        Return the daily cases
        '''
        accumulativeCases = self.newCasesCountries(countries)
        dailyCases = accumulativeCases.drop(['Province/State'],axis=1).diff(axis=1)
        return dailyCases

    def newCasesByCountry(self, country, method = ''):
        '''Design to return the new daily cases from a country, where 
        is considered the sum of all the states.
        '''
        confirmed_cases = self.confirmed.loc[country]
        if type(confirmed_cases) == pd.core.frame.DataFrame:
            print('There are multiple states in {c}'.format(c=country))
            if method.lower() == 'nan':
                confirmed_cases = confirmed_cases[confirmed_cases['Province/State'].isna()]
            confirmed_cases = confirmed_cases.drop(columns=['Province/State']).sum()
        dailyCases = confirmed_cases.diff()
        
        return dailyCases

