import wrds
import pandas as pd
import os
import time
import datetime

class myData:

  def __init__(self):
    SP500List, today_date, year_ago_date, thisyear, lastyear

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
    membership_columns = ['audit_membership', 'cg_membership', 'comp_membership', 'nom_membership']
    total_df = dataframe.groupby('ticker')[membership_columns].apply(lambda x: (x.notna().sum() > 0).sum()).reset_index(name='total_memberships')
    return total_df


  #Has multiple companies which have more than one CEO shown.
  def is_CEO_Dual():
    dataframe = pd.DataFrame(data=(db.raw_sql(f"SELECT TICKER, EMPLOYMENT_CEO, EMPLOYMENT_CHAIRMAN FROM risk.rmdirectors WHERE YEAR BETWEEN '{lastyear}' AND '{thisyear}' AND TICKER IN {SP500List}")))
    #dataframe = pd.DataFrame(data=(db.raw_sql(f"SELECT TICKER, EMPLOYMENT_CEO, EMPLOYMENT_CHAIRMAN, YEAR, meetingdate, NAME, FULLNAME FROM risk.rmdirectors WHERE YEAR BETWEEN '{lastyear}' AND '{thisyear}' AND TICKER IN {SP500List}")))
    com_data = []
    for index, row in dataframe.iterrows():
      if row['employment_ceo'] == 'Yes':
        if row['employment_chairman'] == 'Yes':
          com_data.append({'ticker': row['ticker'], 'CEODuality': True})
        else:
          com_data.append({'ticker': row['ticker'], 'CEODuality': False})
    return pd.DataFrame(com_data)


  def director_power():
    #%Independent Directors & Shares held & Number of committees & Voting type
    dataframe = pd.DataFrame(data=(db.raw_sql(f"SELECT TICKER, CLASSIFICATION, NUM_OF_SHARES, OWNLESS1, PCNT_CTRL_VOTINGPOWER FROM risk.rmdirectors WHERE YEAR BETWEEN '{lastyear}' AND '{thisyear}' AND TICKER IN {SP500List}")))
    print(dataframe)

    
    #Count if any directors have above 4.5% share individually. -- I need total company shares for a %.. find on WRDS somewhere
    #Count percentage of company the board holds. - NUM_OF_SHARES


    #Count if any directors have above ...% voting power. Director holds <1% Voting Power -- is ownless -- could use either
    voting_power = dataframe.groupby('ticker')['pcnt_ctrl_votingpower'].apply(lambda x: (x >= 10).sum()).reset_index(name='high_voting_power')

    #Count number of independent directors using CLASSIFICATION and make a percentage. -- Board affiliation (E-employee/insider; I-Independent; L-linked; NA-not ascertainable) (classification) -- I-NED = Independent Non-Exec director
    num_independent_directors = dataframe.groupby('ticker')['classification'].apply(lambda x: round((x == ('I-NED' or 'I' or 'NI-NED')).sum() / len(x) *100, 1)).reset_index(name='percentage_NEDs')

    



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

    
    #Only has 84 rows instead of 500...
    OrgSummary = pd.DataFrame(data=(db.raw_sql(f"SELECT Ticker, NumberDirectors, GenderRatio, NationalityMix, Annualreportdate FROM boardex.na_wrds_org_summary WHERE Annualreportdate BETWEEN '{year_ago_date}' AND '{today_date}' AND Ticker IN {SP500List}")))
    newSummary = OrgSummary.drop_duplicates()
    #print(newSummary)
    
  
    return merged_data


  def combine_data(dataframe):
    committees = myData.count_committees()
    ceo = myData.is_CEO_Dual()
    director_powerful = myData.director_power()
    ceo_and_committees = pd.merge(committees, ceo, on='ticker', how='inner')
    total_dataset = pd.merge(ceo_and_committees, dataframe, on='ticker', how='inner')
    return total_dataset


if __name__ == "__main__":
  start = time.time()
  db = wrds.Connection(wrds_username="twhittome")

  SP500List = myData.get_SP500_companies()
  today_date, year_ago_date, thisyear, lastyear = myData.get_dates()
  dataframe = myData.read_in_data_from_wrds()
  final_dataset = myData.combine_data(dataframe)

  #output_excel_file(final_dataset, 'orgstaff1.xlsx')

  end = time.time()
  print("The time of execution of above program is :",
        (end-start) * 10**3, "ms")


