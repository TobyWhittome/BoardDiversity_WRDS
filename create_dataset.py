import wrds
import pandas as pd
import os
import time
import datetime


start = time.time()

db = wrds.Connection(wrds_username="twhittome")

def get_dates():
  today = datetime.date.today()
  thisYear = int(today.strftime("%Y"))
  yearMod = thisYear - 1
  modified_date = today.replace(year=yearMod)
  return today, modified_date, thisYear, yearMod


def get_SP500_companies() -> pd.DataFrame:
  url = 'https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx'
  frame = pd.read_excel(url, engine='openpyxl', usecols=['Ticker'], skiprows=4).dropna()
  return tuple(frame['Ticker'])


def output_excel_file(database, filename):
  excel_file_path = os.path.join(os.getcwd(), filename)
  database.to_excel(excel_file_path, index=False)


def count_committees():
  dataframe = pd.DataFrame(data=(db.raw_sql(f"SELECT TICKER, AUDIT_MEMBERSHIP, CG_MEMBERSHIP, COMP_MEMBERSHIP, NOM_MEMBERSHIP FROM risk.rmdirectors WHERE YEAR BETWEEN '{lastyear}' AND '{thisyear}' AND TICKER IN {SP500List}")))
  companyCounted = set()
  com_data = []
  for index, row in dataframe.iterrows():
    if row['ticker'] not in companyCounted:
      count = sum(1 for membership in [row['audit_membership'], row['cg_membership'], row['comp_membership'], row['nom_membership']] if membership is not None)
      com_data.append({'ticker': row['ticker'], 'NumCommittees': count})
      companyCounted.add(row['ticker'])
  return pd.DataFrame(com_data)


def is_CEO_Dual():
  dataframe = pd.DataFrame(data=(db.raw_sql(f"SELECT TICKER, EMPLOYMENT_CEO, EMPLOYMENT_CHAIRMAN FROM risk.rmdirectors WHERE YEAR BETWEEN '{lastyear}' AND '{thisyear}' AND TICKER IN {SP500List}")))
  com_data = []
  for index, row in dataframe.iterrows():
    if row['employment_ceo'] == 'Yes':
      if row['employment_chairman'] == 'Yes':
        com_data.append({'ticker': row['ticker'], 'CEODuality': True})
      else:
        com_data.append({'ticker': row['ticker'], 'CEODuality': False})
  print(pd.DataFrame(com_data))
  return pd.DataFrame(com_data)
                

def read_in_data_from_wrds():

  #Company Database
  query = f'''
  SELECT g.TICKER, g.YEAR, g.DUALCLASS, e.NUMMTGS, e.YEAR
  FROM risk.rmgovernance g
  JOIN comp_execucomp.codirfin e 
  ON g.TICKER = e.TICKER AND e.YEAR = g.YEAR
  WHERE g.YEAR BETWEEN '{lastyear}' AND '{thisyear}' AND g.TICKER IN {SP500List}
  '''
  merged_data = pd.DataFrame(data=(db.raw_sql(query)))
  #print(merged_data)


  #Size (org summary) & Diversity ... need to add Annualreportdate BETWEEN 2023-01-01 AND 2024-01-15
  OrgSummary = pd.DataFrame(data=(db.raw_sql(f"SELECT Ticker, NumberDirectors, GenderRatio, NationalityMix, Annualreportdate FROM boardex.na_wrds_org_summary WHERE Annualreportdate BETWEEN '{year_ago_date}' AND '{today_date}' AND TICKER IN {SP500List}")))
  #print(OrgSummary)

  

  #%Independent Directors & Shares held & CEO Duality & Number of committees & Voting type
  DirectorsUS = pd.DataFrame(data=(db.raw_sql(f"SELECT TICKER, CLASSIFICATION, NUM_OF_SHARES, EMPLOYMENT_CEO, EMPLOYMENT_CHAIRMAN, OWNLESS1, PCNT_CTRL_VOTINGPOWER FROM risk.rmdirectors WHERE YEAR BETWEEN '{lastyear}' AND '{thisyear}' AND TICKER IN {SP500List}")))




SP500List = get_SP500_companies()
today_date, year_ago_date, thisyear, lastyear = get_dates()
count_committees()
ceo = is_CEO_Dual()
read_in_data_from_wrds()
#output_excel_file(merged_data, 'mergedata.xlsx')
#output_excel_file(ceo, 'ceoDuality.xlsx')
dataframe = pd.DataFrame(data=(db.raw_sql(f"SELECT TICKER, EMPLOYMENT_CEO, EMPLOYMENT_CHAIRMAN, YEAR, meetingdate, NAME, FULLNAME FROM risk.rmdirectors WHERE YEAR BETWEEN '{lastyear}' AND '{thisyear}' AND TICKER IN {SP500List}")))
print(dataframe)
output_excel_file(dataframe, 'ceoDuality2.xlsx')


end = time.time()
print("The time of execution of above program is :",
      (end-start) * 10**3, "ms")