import pandas as pd

from datetime import datetime, timedelta

# nz = c.loc[c[c.columns[1]] == 'New Zealand']
# nz[nz.columns[-1]].iloc[-1] - nz[nz.columns[-2]].iloc[-1]
# nz.iloc[-1,-1] - nz.iloc[-1,-2]

class CovidDB:
    def __init__(self):
        url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
        url_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
        url_recovered = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
        self.confirmed = pd.read_csv(url_confirmed)
        self.deaths = pd.read_csv(url_deaths)
        self.recovered = pd.read_csv(url_recovered)
        self.date_format = '%m/%d/%y'
        
    def countryCases(self, country, date):
        index = 1
        today = datetime.strptime(date, self.date_format)
        yesterday = today - timedelta(days=1)
        lastDate = yesterday.strftime(self.date_format)
        if lastDate[0] == '0':
            lastDate = lastDate[1:]
        
        cases = self.confirmed.loc[self.confirmed[self.confirmed.columns[index]] == country]
        dailyCases = cases[date] - cases[lastDate]
        print('Confirmed cases in {country} between {lastDate} and {date}:'.format(country=country,lastDate=lastDate,date=date))
        print(dailyCases,'\n')
        return dailyCases

