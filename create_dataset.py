import wrds
import pandas as pd
import os
import time
import datetime
from dateutil.relativedelta import relativedelta

class myData:

  def __init__(self):
    self.SP500Tickers, self.today_date, self.year_ago_date, self.month_ago_date, self.thisyear, self.lastyear, self.db, self.SP500IDs, self.SP500_permo = None, None, None, None, None, None, None, None, None

  def get_dates(self):
    today = datetime.date.today()
    one_month_ago = today - relativedelta(months=1)
    thisYear = int(today.strftime("%Y"))
    yearMod = thisYear - 1
    modified_date = today.replace(year=yearMod)
    return today, modified_date, thisYear, yearMod, one_month_ago


  def get_SP500_companies(self) -> pd.DataFrame:
    url = 'https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx'
    frame = pd.read_excel(url, engine='openpyxl', usecols=['Ticker'], skiprows=4).dropna()
    return tuple(frame['Ticker'])
  
  def get_SP500_IDs(self):
    # need to change
    frame = pd.DataFrame(data=(self.db.raw_sql(f"SELECT DISTINCT boardid FROM boardex.na_wrds_company_profile WHERE ticker IN {self.SP500Tickers}"))) # is the issue that some boardIds overlap?
    print(len(frame))
    return tuple(frame['boardid'])
  
  def get_SP500_permno(self):
    #Has duplicate IDs
    id_ticker = pd.DataFrame(data=(self.db.raw_sql(f"SELECT DISTINCT ticker, MIN(boardid) AS boardid FROM boardex.na_wrds_company_profile WHERE ticker IN {self.SP500Tickers} GROUP BY ticker")))
    #print(id_ticker)
    IDs = tuple(id_ticker['boardid'])
    #print(len(IDs))

    #can't get this to work until SP500 IDs pulls in the right ones.  
    id_permco = pd.DataFrame(data=(self.db.raw_sql(f"SELECT DISTINCT permco, companyid FROM wrdsapps.bdxcrspcomplink WHERE companyid IN {IDs}")))
    duplicated_rows = id_permco[id_permco.duplicated(subset=["companyid"], keep=False)]
    sorted_duplicated_rows = duplicated_rows.sort_values(by='companyid', ascending=True) 
    self.output_excel_file(sorted_duplicated_rows, 'permcoduplication1.xlsx')

    id_permco.rename(columns={'companyid': 'boardid'}, inplace=True)
    complete_frame = (pd.merge(id_ticker, id_permco, on='boardid', how='inner'))
    return complete_frame
    
  
  def output_excel_file(self, database, filename):
    excel_file_path = os.path.join(os.getcwd(), filename)
    database.to_excel(excel_file_path, index=False)


  def count_committees(self):
    dataframe = pd.DataFrame(data=(self.db.raw_sql(f"SELECT TICKER, AUDIT_MEMBERSHIP, CG_MEMBERSHIP, COMP_MEMBERSHIP, NOM_MEMBERSHIP, YEAR, CUSIP FROM risk.rmdirectors WHERE YEAR BETWEEN '{self.lastyear}' AND '{self.thisyear}' AND TICKER IN {self.SP500Tickers}")))

    boardsize = dataframe.groupby('ticker').size().reset_index(name='boardsize')
    membership_columns = ['audit_membership', 'cg_membership', 'comp_membership', 'nom_membership']
    total_memberships = dataframe.groupby('ticker')[membership_columns].apply(lambda x: (x.notna().sum() > 0).sum()).reset_index(name='total_memberships')

    return pd.merge(total_memberships, boardsize, on='ticker', how='inner')


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

  #def gender_ratio(self):
    # OrgSummary = pd.DataFrame(data=(self.db.raw_sql(f"SELECT Ticker, NumberDirectors, GenderRatio, NationalityMix, Annualreportdate FROM boardex.na_wrds_org_summary WHERE Annualreportdate BETWEEN '{self.year_ago_date}' AND '{self.today_date}' AND Ticker IN {self.SP500Tickers}")))


  def director_power(self):
    
    #1. Count if any directors have above 4.5% share individually.
    #2. TotalOf (All Directors share %. = NUM_Shares / Outstanding)
    
    #Outstanding shares Compustat
    outstanding_shares = pd.DataFrame(data=(self.db.raw_sql(f"SELECT TIC, CSHO FROM comp_na_daily_all.funda WHERE FYEAR BETWEEN '2022' AND '{self.thisyear}' AND TIC IN {self.SP500Tickers}")))
    fixed_outstanding = outstanding_shares.dropna().drop_duplicates(subset=['tic'], keep='last').reset_index(drop=True)
    
    #%Independent Directors & Shares held & Number of committees & Voting type
    dataframe = pd.DataFrame(data=(self.db.raw_sql(f"SELECT TICKER, CLASSIFICATION, PCNT_CTRL_VOTINGPOWER FROM risk.rmdirectors WHERE YEAR BETWEEN '{self.lastyear}' AND '{self.thisyear}' AND TICKER IN {self.SP500Tickers}")))
    
    #Directors indivdual share %
    num_shares_df = pd.DataFrame(data=(self.db.raw_sql(f"SELECT TICKER, NUM_OF_SHARES FROM risk.rmdirectors WHERE YEAR BETWEEN '{self.lastyear}' AND '{self.thisyear}' AND TICKER IN {self.SP500Tickers}")))
    shares_csho_merged = pd.merge(fixed_outstanding, num_shares_df, how='inner', left_on='tic', right_on='ticker')
    shares_csho_merged['indiv_share_%'] = round((shares_csho_merged['num_of_shares'] / (shares_csho_merged['csho']*1000000)) * 100 , 3)
    
    #If the directors have above 4.5% then count 1 and return total
    shares_csho_merged['numof_>4.5%_shareholders'] = shares_csho_merged['indiv_share_%'].count()


    result_df = dataframe.groupby('ticker').agg(
    high_voting_power=('pcnt_ctrl_votingpower', lambda x: (x >= 10).sum()),
    percentage_INEDs=('classification', lambda x: round((x.isin(['I-NED', 'I', 'NI-NED'])).mean() * 100, 1))
    ).reset_index()

    result_df['indiv_share_%'] = shares_csho_merged['indiv_share_%']
    
    #Directors total share %
    total_share = shares_csho_merged.groupby('tic')['indiv_share_%'].sum().reset_index(name='total_share_%')
    final_df = pd.concat([result_df, total_share], axis=1)
    final_df.drop(columns=['tic', 'indiv_share_%'], inplace=True)
    
    return final_df

  
  def dualclass(self):

    query = f'''
    SELECT g.TICKER, g.DUALCLASS
    FROM risk.rmgovernance g
    JOIN comp_execucomp.codirfin e 
    ON g.TICKER = e.TICKER AND e.YEAR = g.YEAR
    WHERE g.YEAR BETWEEN '{self.lastyear}' AND '{self.thisyear}' AND g.TICKER IN {self.SP500Tickers}
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
    total_dataset = pd.merge(pd.merge(pd.merge(director_powerful, committees, on='ticker', how='inner'), ceo, on='ticker', how='inner'), dualclass, on='ticker', how='inner')
    return total_dataset



def main():
  start = time.time()
  inst = myData()
  inst.db = wrds.Connection(wrds_username="twhittome")
  inst.SP500Tickers = inst.get_SP500_companies()
  inst.SP500IDs = inst.get_SP500_IDs()
  inst.SP500_permo = inst.get_SP500_permno()
  inst.today_date, inst.year_ago_date, inst.thisyear, inst.lastyear, inst.month_ago_date = inst.get_dates()

  final_dataset = inst.combine_data()
  #inst.output_excel_file(final_dataset, 'orgstaff1.xlsx')

  end = time.time()
  print("The time of execution of above program is :",
        (end-start) * 10**3, "ms")
  
  return final_dataset


if __name__ == "__main__":
  #print(main())
  main()


