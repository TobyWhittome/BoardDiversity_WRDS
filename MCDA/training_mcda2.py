#This file will take results from the past 10 years to calculate the MCDA scores for each company. 
# The idea being that when new data comes in every year, the MCDA scores will be updated and the companies will be ranked accordingly.

import pandas as pd
import datetime
import MCDA as mcda

#for every step of 3 years, average the TOPSIS scores from each company to create a singular score

today = datetime.date.today()
thisYear = int(today.strftime("%Y"))


for year in range(thisYear, 2007, -3):
  
  new_df = pd.DataFrame(columns=['ticker', 'Topsis_score', 'year'])

  yearname = f'{year}_dataset'
  df = pd.read_excel(f'prev_data/{yearname}.xlsx')

  yearname2 = f'{year-1}_dataset'
  df2 = pd.read_excel(f'prev_data/{yearname2}.xlsx')

  yearname3 = f'{year-2}_dataset'
  df3 = pd.read_excel(f'prev_data/{yearname3}.xlsx')

  """  print(f"Year {year}")
  print(df)
  print(f'Year {year-1}')
  print(df2)
  print(f'Year {year-2}')
  print(df3) """


  #Perform MCDA on all of them
  mcda_df = mcda.main(df, [])
  mcda_df2 = mcda.main(df2, [])
  mcda_df3 = mcda.main(df3, [])


  #average the TOPSIS scores from the past 3 years
  new_df['Topsis Score'] = (mcda_df['Topsis Score'] + mcda_df2['Topsis Score'] + mcda_df3['Topsis Score']) / 3
  new_df['ticker'] = df['ticker']
  new_df['year'] = year

  print(new_df)

