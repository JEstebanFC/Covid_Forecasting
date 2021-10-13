
from optparse import OptionParser

from Utils.CovidDB import CovidDB
from Models import DATA_PATH_NEW, RESULTS_PATH


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-c', '--country', dest='country', default='')
    parser.add_option('-s', '--state', dest='state', default='')
    parser.add_option('-d', '--date', dest='date', default='')
    options, args = parser.parse_args()

    options.date = '8/29/21'
    options.country = 'New Zealand,Australia'
    options.country = options.country.split(',')
    
    covidDB = CovidDB()
    dailyCases = covidDB.newCasesReport(options.country)
    fileName = DATA_PATH_NEW + 'AllCases.csv'
    print('Data saved in:', fileName)
    dailyCases.to_csv(fileName)
    # for country in options.country:
    #     newCases,deathCases,recoverCases = covidDB.countryCases(country,options.date)
