#This file will take results from the past 10 years to calculate the MCDA scores for each company. 
# The idea being that when new data comes in every year, the MCDA scores will be updated and the companies will be ranked accordingly.

import pandas as pd
import datetime
import MCDA as mcda
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

#for every step of 3 years, average the TOPSIS scores from each company to create a singular score

today = datetime.date.today()
thisYear = int(today.strftime("%Y"))

average_scores_list = []


def average_scores():
  
  for year in range(thisYear - 1, 2010, -3):
    
    new_df = pd.DataFrame(columns=['ticker', 'Topsis Score', 'years'])

    yearname = f'{year}_dataset'
    df_og = pd.read_excel(f'mcda_prev_data/{yearname}.xlsx')

    yearname2 = f'{year-1}_dataset'
    df2_og = pd.read_excel(f'mcda_prev_data/{yearname2}.xlsx')

    yearname3 = f'{year-2}_dataset'
    df3_og = pd.read_excel(f'mcda_prev_data/{yearname3}.xlsx')

    mcda_df, corr = mcda.main(df_og, [])
    mcda_df2, corr = mcda.main(df2_og, [])
    mcda_df3, corr = mcda.main(df3_og, [])
    
    
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
            zeros_df = pd.DataFrame(0, index=range(largest - len(df)), columns=df.columns)
            modified_df = pd.concat([df, zeros_df]).reset_index(drop=True)
            modified_dfs.append(modified_df)  
        else:
            modified_dfs.append(df)  

    mcda_df, mcda_df2, mcda_df3 = modified_dfs

            
    divisor = ((mcda_df['Topsis Score'] != 0).astype(int) +
            (mcda_df2['Topsis Score'] != 0).astype(int) +
            (mcda_df3['Topsis Score'] != 0).astype(int))

    #average the TOPSIS scores from the past 3 years
    new_df['Topsis Score'] = (mcda_df['Topsis Score'] + mcda_df2['Topsis Score'] + mcda_df3['Topsis Score']) / divisor
    new_df['ticker'] = largest_db['ticker']
    new_df['years'] = f'{year-2}-{year}'
    new_df['tobinsQ'] = (mcda_df['tobinsQ'] + mcda_df2['tobinsQ'] + mcda_df3['tobinsQ']) / divisor
    average_scores_list.append(new_df)
    
  return average_scores_list



def visualize(average_scores_list):
  
  #Plot correlations against years
  correlations = []
  yearz = []
  print(average_scores_list)
  for df in average_scores_list:
    df.dropna(inplace=True)
    correlationspear, _ = stats.spearmanr(df['Topsis Score'], df['tobinsQ'])
    yearz.append(df['years'].iloc[0])
    correlations.append(correlationspear)
    
  
    #Just get one value from the years column
    # Assuming df is your DataFrame and 'years' is a column in it
    #Its because the 2024 data is empty
    if not df['years'].empty:
        years = df['years'].iloc[0]
        print(years)
    else:
        print("The 'years' column is empty.")
        years = '2024'
        

    print(f"Spearman's rank correlation for year {years}: {correlationspear} \n")
    #sns.set_theme()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Topsis Score', y='tobinsQ', data=df, color='b')
    plt.title(f'{years}', fontweight='bold', fontsize=16)
    plt.xlabel('Topsis Score', fontweight='bold', fontsize=16)
    plt.ylabel('Tobins Q', fontweight='bold', fontsize=16)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.ylim(-0.5, 15)
    
  
  
  plt.figure(figsize=(10, 6))
  plt.plot(yearz, correlations, color='blue', linewidth=3)
  plt.xlabel('Year Groups', fontweight='bold', fontsize=16)
  plt.ylabel('Spearman\'s Rank', fontweight='bold', fontsize=16)
  plt.xticks(fontsize=14, fontweight='bold')
  plt.yticks(fontsize=14, fontweight='bold')
  plt.ylim(-0.5, 0.5)
  plt.grid(True)
  plt.gca().invert_xaxis()
  plt.show()
  

if __name__ == "__main__":
  average_scores_list = average_scores()
  visualize(average_scores_list)


