import wrds
import pandas as pd
import os
import time
import datetime
from dateutil.relativedelta import relativedelta
from functools import reduce

class myData:

  def __init__(self):
    self.SP500Tickers, self.today_date, self.year_ago_date, self.month_ago_date, self.thisyear, self.lastyear, self.twoyearago, self.db, self.SP500IDs = None, None, None, None, None, None, None, None, None

  def get_dates(self):
    today = datetime.date.today()
    one_month_ago = today - relativedelta(months=1)
    thisYear = int(today.strftime("%Y")) - 1
    lastyear = thisYear - 1
    modified_date = today.replace(year=lastyear)
    twoago = lastyear - 1
    return today, modified_date, thisYear, lastyear, one_month_ago, twoago
  
  def set_dates(self, year):
    thisYear = year
    lastyear = thisYear - 1
    today_date = datetime.date.today().replace(year=thisYear)
    year_ago_date = today_date.replace(year=lastyear)
    one_month_ago = today_date - relativedelta(months=1)
    twoago = lastyear - 1
    return today_date, year_ago_date, thisYear, lastyear, one_month_ago, twoago


  def get_SP500_companies(self) -> pd.DataFrame:
    url = 'https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx'
    frame = pd.read_excel(url, engine='openpyxl', usecols=['Ticker'], skiprows=4).dropna()
    return tuple(frame['Ticker'])
  
             
  def get_hist_SP500_companies(self) -> tuple:
    combined_query = f"""
    SELECT c.tic as ticker
    FROM crsp_a_ccm.ccm_lookup c
    INNER JOIN crsp_a_indexes.dsp500list d
    ON c.lpermno = d.permno
    WHERE d.start <= '{self.today_date}' AND d.ending >= '{self.year_ago_date}'
    """
    result = self.db.raw_sql(combined_query)
    return tuple(result['ticker'].drop_duplicates().reset_index(drop=True))

  
  def get_SP500_IDs(self):
    id_ticker = pd.DataFrame(data=(self.db.raw_sql(f"SELECT DISTINCT ticker, MIN(boardid) AS boardid FROM boardex.na_wrds_company_profile WHERE ticker IN {self.SP500Tickers} GROUP BY ticker")))
    return tuple(id_ticker['boardid'])
    
  
  def output_excel_file(self, database, filename):
    excel_file_path = os.path.join(os.getcwd(), filename)
    database.to_excel(excel_file_path, index=False)


  def count_committees(self):
    dataframe = pd.DataFrame(data=(self.db.raw_sql(f"SELECT TICKER, AUDIT_MEMBERSHIP, CG_MEMBERSHIP, COMP_MEMBERSHIP, NOM_MEMBERSHIP, YEAR, CUSIP FROM risk.rmdirectors WHERE YEAR BETWEEN '{self.lastyear}' AND '{self.thisyear}' AND TICKER IN {self.SP500Tickers}")))

    boardsize = dataframe.groupby('ticker').size().reset_index(name='boardsize')

    #Not using number of committees anymore
    #membership_columns = ['audit_membership', 'cg_membership', 'comp_membership', 'nom_membership']
    #total_memberships = dataframe.groupby('ticker')[membership_columns].apply(lambda x: (x.notna().sum() > 0).sum()).reset_index(name='total_memberships')
    #return pd.merge(total_memberships, boardsize, on='ticker', how='inner')

    return boardsize


  def is_CEO_Dual(self):
    # % should be roughly 44%, so 327/790 = 41% -- is good
    dataframe = pd.DataFrame(data=(self.db.raw_sql(f"SELECT TICKER, EMPLOYMENT_CEO, EMPLOYMENT_CHAIRMAN FROM risk.rmdirectors WHERE YEAR BETWEEN '{self.lastyear}' AND '{self.thisyear}' AND TICKER IN {self.SP500Tickers}")))
    com_data = []
    for index, row in dataframe.iterrows():
      if row['employment_ceo'] == 'Yes':
        if row['employment_chairman'] == 'Yes':
          com_data.append({'ticker': row['ticker'], 'CEODuality': 1})
        else:
          com_data.append({'ticker': row['ticker'], 'CEODuality': 0})
                  
    new_com_data = pd.DataFrame(com_data).drop_duplicates().reset_index(drop=True)
          
    rows_to_delete = []
    for index, row in new_com_data.iterrows():
      if index < len(new_com_data) - 1:  # Ensure we don't go out of bounds
          next_row = new_com_data.iloc[index + 1]          
          if row['ticker'] == next_row['ticker']:
              if row['CEODuality'] == 1:           
                  rows_to_delete.append(next_row.name)
              elif next_row['CEODuality'] == 1:                 
                  rows_to_delete.append(row.name)
                  
    new_com_data.drop(rows_to_delete, inplace=True)
    new_com_data.reset_index(drop=True)
    return new_com_data

  def gender_ratio(self):
    two_year_ago_date = self.today_date.replace(year=self.lastyear - 1)
    #two_year_ago_date = datetime.date.today().replace(year=2022)
    OrgSummary = pd.DataFrame(data=(self.db.raw_sql(f"SELECT Ticker, NumberDirectors, GenderRatio, NationalityMix, Annualreportdate FROM boardex.na_wrds_org_summary WHERE Annualreportdate BETWEEN '{two_year_ago_date}' AND '{self.today_date}' AND Ticker IN {self.SP500Tickers}")))
    
    sorted_OrgSummary = OrgSummary.dropna().sort_values(by=['ticker', 'annualreportdate'], ascending=[True, False])
    fixed_Orgsummary = sorted_OrgSummary.drop_duplicates(subset=['ticker'], keep='first').reset_index(drop=True)
 
    fixed_Orgsummary.drop(columns=['annualreportdate', 'numberdirectors'], inplace=True)
    return fixed_Orgsummary
    

  def tobinsQ(self):
    data = pd.DataFrame(data=(self.db.raw_sql(f"SELECT TIC, MKVALT, AT FROM comp_na_daily_all.funda WHERE FYEAR BETWEEN '{self.lastyear}' AND '{self.thisyear}' AND TIC IN {self.SP500Tickers}")))
    fixed_outstanding = data.dropna().drop_duplicates(subset=['tic'], keep='last').reset_index(drop=True)
    fixed_outstanding.rename(columns={'tic': 'ticker'}, inplace=True)

    fixed_outstanding['tobinsQ'] = fixed_outstanding['mkvalt'] / fixed_outstanding['at']

    #gives 2.08 average which should be 1.4 according to today's stats.
    print(fixed_outstanding['tobinsQ'].mean())

    #returning only ticker and tobinsQ
    return fixed_outstanding[['ticker', 'tobinsQ']]
  


  def director_power(self):
    
    #Outstanding shares Compustat
    outstanding_shares = pd.DataFrame(data=(self.db.raw_sql(f"SELECT TIC, CSHO FROM comp_na_daily_all.funda WHERE FYEAR BETWEEN '{self.twoyearago}' AND '{self.thisyear}' AND TIC IN {self.SP500Tickers}")))
    fixed_outstanding = outstanding_shares.dropna().drop_duplicates(subset=['tic'], keep='last').reset_index(drop=True)
    
    #%Independent Directors & Shares held & Number of committees & Voting type
    dataframe = pd.DataFrame(data=(self.db.raw_sql(f"SELECT TICKER, CLASSIFICATION, PCNT_CTRL_VOTINGPOWER FROM risk.rmdirectors WHERE YEAR BETWEEN '{self.lastyear}' AND '{self.thisyear}' AND TICKER IN {self.SP500Tickers}")))
    
    #Directors indivdual share %
    num_shares_df = pd.DataFrame(data=(self.db.raw_sql(f"SELECT TICKER, NUM_OF_SHARES FROM risk.rmdirectors WHERE YEAR BETWEEN '{self.lastyear}' AND '{self.thisyear}' AND TICKER IN {self.SP500Tickers}")))
    shares_csho_merged = pd.merge(fixed_outstanding, num_shares_df, how='inner', left_on='tic', right_on='ticker')
    shares_csho_merged['indiv_share_%'] = round((shares_csho_merged['num_of_shares'] / (shares_csho_merged['csho']*1000000)) * 100 , 3)
    
    #If the directors have above 4.5% then count 1 and return total
    num_directors_45 = (shares_csho_merged.groupby('tic')['indiv_share_%']
    .apply(lambda x: ((x > 4.5).sum() if (x > 4.5).any() else 0))
    .reset_index(name='num_directors_>4.5'))

    result_df = dataframe.groupby('ticker').agg(
    #high_voting_power=('pcnt_ctrl_votingpower', lambda x: (x >= 10).sum()),
      
    #voting power is the percentage directors with a voting power greater than 0
    voting_power=('pcnt_ctrl_votingpower', lambda x: round(((x > 0).mean() * 100), 1)),
    percentage_INEDs=('classification', lambda x: round((x.isin(['I-NED', 'I', 'NI-NED'])).mean() * 100, 1))
    ).reset_index()
    
    #Directors >4.5
    result_df['num_directors_>4.5'] = num_directors_45['num_directors_>4.5']
    
    #Directors total share %
    total_share = shares_csho_merged.groupby('tic')['indiv_share_%'].sum().reset_index(name='total_share_%')
    final_df = pd.concat([result_df, total_share], axis=1)
    final_df.drop(columns=['tic'], inplace=True)
    
    return final_df
  
  
    #Not using this anymore
    """   def market_cap(self):
    #In millions
    mcap = pd.DataFrame(data=(self.db.raw_sql(f"SELECT DISTINCT ticker, MAX(mktcapitalisation) as mktcapitalisation FROM boardex.na_wrds_company_profile WHERE ticker IN {self.SP500Tickers} GROUP BY ticker")))
    mcap = mcap.dropna().reset_index(drop=True)
    return mcap """

  
  def dualclass(self):

    query = f'''
    SELECT g.TICKER, g.DUALCLASS
    FROM risk.rmgovernance g
    JOIN comp_execucomp.codirfin e 
    ON g.TICKER = e.TICKER AND e.YEAR = g.YEAR
    WHERE g.YEAR = {self.thisyear} AND g.TICKER IN {self.SP500Tickers}
    '''
    dataframe = pd.DataFrame(data=(self.db.raw_sql(query)))
    dataframe['dualclass'].replace(to_replace='YES', value=1, inplace=True)
    dataframe['dualclass'].fillna(0, inplace=True)
    dataframe['dualclass'] = dataframe['dualclass'].astype(int)
    
    return dataframe

  def combine_data(self):
    dualclass = self.dualclass()
    committees = self.count_committees()
    ceo = self.is_CEO_Dual()
    director_powerful = self.director_power()
    #mcap = self.market_cap()
    tobinsQ = self.tobinsQ()
    genderRatio = self.gender_ratio()
    
    #print(len(dualclass), len(committees), len(ceo), len(director_powerful), len(tobinsQ), len(genderRatio), self.thisyear)

    dfs = [genderRatio, director_powerful, committees, tobinsQ, ceo, dualclass]
    total_dataset = reduce(lambda left, right: pd.merge(left, right, on='ticker', how='inner'), dfs)

    final = total_dataset.drop_duplicates().reset_index(drop=True)
    final.dropna(inplace=True)
    return final


def past_data(year, conn):
  
  inst = myData()
  inst.db = conn
  inst.today_date, inst.year_ago_date, inst.thisyear, inst.lastyear, inst.month_ago_date, inst.twoyearago = inst.set_dates(year)
  inst.SP500Tickers = inst.get_hist_SP500_companies()
  inst.SP500IDs = inst.get_SP500_IDs()
  final_dataset = inst.combine_data()
  
  return final_dataset
  

def main():
  start = time.time()
  inst = myData()
  inst.db = wrds.Connection(wrds_username="twhittome")
  inst.today_date, inst.year_ago_date, inst.thisyear, inst.lastyear, inst.month_ago_date, inst.twoyearago = inst.get_dates()
  inst.SP500Tickers = inst.get_hist_SP500_companies()
  inst.SP500IDs = inst.get_SP500_IDs()
  final_dataset = inst.combine_data()
  final_dataset.dropna(inplace=True)
  inst.output_excel_file(final_dataset, 'dataset/final_dataset.xlsx')

  end = time.time()
  print("The time of execution of above program is :",
        round((end-start),1), "s")
  
  return final_dataset


if __name__ == "__main__":
  main()



