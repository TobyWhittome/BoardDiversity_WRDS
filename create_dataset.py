import wrds
import pandas as pd
import os
import time
import datetime

class myData:

  def __init__(self):
    self.SP500Tickers, self.today_date, self.year_ago_date, self.thisyear, self.lastyear, self.db, self.SP500IDs = None, None, None, None, None, None, None

  def get_dates(self):
    today = datetime.date.today()
    thisYear = int(today.strftime("%Y"))
    yearMod = thisYear - 1
    modified_date = today.replace(year=yearMod)
    return today, modified_date, thisYear, yearMod


  def get_SP500_companies(self) -> pd.DataFrame:
    url = 'https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx'
    frame = pd.read_excel(url, engine='openpyxl', usecols=['Ticker'], skiprows=4).dropna()
    return tuple(frame['Ticker'])
  
  def get_SP500_IDs(self):
    frame = pd.DataFrame(data=(self.db.raw_sql(f"SELECT DISTINCT boardid FROM boardex.na_wrds_company_profile WHERE ticker IN {self.SP500Tickers}")))
    return tuple(frame['boardid'])


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
    #%Independent Directors & Shares held & Number of committees & Voting type
    dataframe = pd.DataFrame(data=(self.db.raw_sql(f"SELECT TICKER, CLASSIFICATION, NUM_OF_SHARES, OWNLESS1, PCNT_CTRL_VOTINGPOWER FROM risk.rmdirectors WHERE YEAR BETWEEN '{self.lastyear}' AND '{self.thisyear}' AND TICKER IN {self.SP500Tickers}")))
    #print(dataframe)

    #Legacy data unfort
    dataframe2 = pd.DataFrame(data=(self.db.raw_sql(f"SELECT Ticker, sumaflin FROM block.block WHERE TICKER IN {self.SP500Tickers}")))
    #print(dataframe2)


    #Count if any directors have above 4.5% share individually. -- I need total company shares for a %..
    #Count percentage of company the board holds. - NUM_OF_SHARES

    dataframe3 = pd.DataFrame(data=(self.db.raw_sql(f"SELECT TICKER, SHROWN_TOT_PCT, SHROWN_EXCL_OPTS_PCT, EXEC_FULLNAME FROM comp_execucomp.anncomp WHERE YEAR BETWEEN '{self.lastyear}' AND '{self.thisyear}' AND TICKER IN {self.SP500Tickers}")))
    #print(dataframe3)
    self.output_excel_file(dataframe3, 'sharestest.xlsx')

    result_df = dataframe.groupby('ticker').agg(

    high_voting_power=('pcnt_ctrl_votingpower', lambda x: (x >= 10).sum()),
    percentage_INEDs=('classification', lambda x: round((x.isin(['I-NED', 'I', 'NI-NED'])).mean() * 100, 1))

    ).reset_index()

    return result_df

  
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
  inst.today_date, inst.year_ago_date, inst.thisyear, inst.lastyear = inst.get_dates()

  final_dataset = inst.combine_data()
  #inst.output_excel_file(final_dataset, 'orgstaff1.xlsx')

  end = time.time()
  print("The time of execution of above program is :",
        (end-start) * 10**3, "ms")
  
  return final_dataset


if __name__ == "__main__":
  print(main())
  #main()


