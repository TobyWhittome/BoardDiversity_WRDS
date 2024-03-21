#This file will take results from the past 10 years to calculate the MCDA scores for each company. 
# The idea being that when new data comes in every year, the MCDA scores will be updated and the companies will be ranked accordingly.

import pandas as pd
import datetime
import MCDA as mcda

#for every step of 3 years, average the TOPSIS scores from each company to create a singular score

today = datetime.date.today()
thisYear = int(today.strftime("%Y"))

average_scores_list = []


def average_scores():
  
  for year in range(thisYear, 2007, -3):
    
    new_df = pd.DataFrame(columns=['ticker', 'Topsis Score', 'years'])

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
    
    
    largest = max(len(mcda_df), len(mcda_df2), len(mcda_df3))
  
    if largest == len(mcda_df):
      largest_db = mcda_df
    elif largest == len(mcda_df2):
        largest_db = mcda_df2
    else:
        largest_db = mcda_df3
        
    dataframes = [mcda_df, mcda_df2, mcda_df3] 
    modified_dfs = [] 

    for df in dataframes:
        if len(df) < largest:
            # Create a DataFrame filled with zeros, with the required number of rows to match 'largest'
            zeros_df = pd.DataFrame(0, index=range(largest - len(df)), columns=df.columns)
            # Use pandas.concat to append the zeros_df to df
            modified_df = pd.concat([df, zeros_df]).reset_index(drop=True)
            modified_dfs.append(modified_df)  # Add the modified dataframe to the list
        else:
            modified_dfs.append(df)  # Add the unmodified dataframe if it's already the largest

    mcda_df, mcda_df2, mcda_df3 = modified_dfs

            
    divisor = ((mcda_df['Topsis Score'] != 0).astype(int) +
            (mcda_df2['Topsis Score'] != 0).astype(int) +
            (mcda_df3['Topsis Score'] != 0).astype(int))

    #average the TOPSIS scores from the past 3 years
    new_df['Topsis Score'] = (mcda_df['Topsis Score'] + mcda_df2['Topsis Score'] + mcda_df3['Topsis Score']) / divisor
    new_df['ticker'] = largest_db['ticker']
    new_df['years'] = f'{year-2}-{year}'
    
    average_scores_list.append(new_df)
    
  return average_scores_list

if __name__ == "__main__":
  average_scores()


