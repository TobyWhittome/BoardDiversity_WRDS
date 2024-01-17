import wrds
import pandas as pd
import os
db = wrds.Connection(wrds_username="twhittome")


def get_SP500_companies() -> pd.DataFrame:
  url = 'https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx'
  frame = pd.read_excel(url, engine='openpyxl', usecols=['Ticker'], skiprows=4).dropna()
  return tuple(frame['Ticker'])


def output_excel_file(database, filename):
  excel_file_path = os.path.join(os.getcwd(), filename)
  database.to_excel(excel_file_path, index=False)


def count_committees(dataframe):
  for index, row in dataframe.iterrows():
      count = 0
      name = row['ticker']
      if (row['audit_membership'] != None):
        count += 1
      if row['cg_membership'] != None:
        count += 1
      if row['comp_membership'] != None:
        count += 1
      if row['nom_membership'] != None:
        count += 1
      

def read_in_data_from_wrds():

  query = '''
  SELECT g.TICKER, g.MEETINGDATE, g.MTGMONTH, g.YEAR, g.DUALCLASS, e.NUMMTGS, e.YEAR
  FROM risk.rmgovernance g
  JOIN comp_execucomp.codirfin e ON g.TICKER = e.TICKER
  '''
  merged_data = pd.DataFrame(data=(db.raw_sql(query)))
  print(merged_data)

  #Size (org summary) & Diversity ... need to add Annualreportdate BETWEEN 2023-01-01 AND 2024-01-15
  OrgSummary = pd.DataFrame(data=(db.raw_sql(f'SELECT Ticker, NumberDirectors, GenderRatio, NationalityMix FROM boardex.na_wrds_org_summary WHERE TICKER IN {SP500List}')))
  #%Independent Directors & Shares held & CEO Duality & Number of committees & Voting type
  DirectorsUS = pd.DataFrame(data=(db.raw_sql(f'SELECT TICKER, CLASSIFICATION, NUM_OF_SHARES, EMPLOYMENT_CEO, EMPLOYMENT_CHAIRMAN, AUDIT_MEMBERSHIP, CG_MEMBERSHIP, COMP_MEMBERSHIP, NOM_MEMBERSHIP, OWNLESS1, PCNT_CTRL_VOTINGPOWER FROM risk.rmdirectors WHERE YEAR BETWEEN 2023 AND 2024 AND TICKER IN {SP500List}')))
  count_committees(DirectorsUS)
  Governance = pd.DataFrame(data=(db.raw_sql('SELECT TICKER, DUALCLASS FROM risk.rmgovernance WHERE YEAR BETWEEN 2023 AND 2024')))
  Execucomp = pd.DataFrame(data=(db.raw_sql('SELECT TICKER, NUMMTGS, YEAR FROM comp_execucomp.codirfin')))


#output_excel_file(DirectorsUS, 'rmdirectors.xlsx')
#output_excel_file(NumCommittees, 'rmdirectors2.xlsx')


SP500List = get_SP500_companies()
read_in_data_from_wrds()