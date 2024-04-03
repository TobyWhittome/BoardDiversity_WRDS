import pandas as pd
import datetime
import MCDA as mcda
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

today = datetime.date.today()
thisYear = int(today.strftime("%Y"))
average_scores_list = []

def average_scores():
  for year in range(thisYear - 1, 2008, -2):
    new_df = pd.DataFrame(columns=['ticker', 'Topsis Score', 'years'])

    yearname = f'{year}_dataset'
    df_og = pd.read_excel(f'prev_data/{yearname}.xlsx')
    yearname2 = f'{year-1}_dataset'
    df2_og = pd.read_excel(f'prev_data/{yearname2}.xlsx')

    mcda_df, corr = mcda.main(df_og, [])
    mcda_df2, corr = mcda.main(df2_og, [])
   
    largest = max(len(mcda_df), len(mcda_df2))
  
    if largest == len(mcda_df):
      largest_db = mcda_df
    else:
        largest_db = mcda_df2
        
    dataframes = [mcda_df, mcda_df2] 
    modified_dfs = [] 

    for df in dataframes:
        if len(df) < largest:       
            zeros_df = pd.DataFrame(0, index=range(largest - len(df)), columns=df.columns)
            modified_df = pd.concat([df, zeros_df]).reset_index(drop=True)
            modified_dfs.append(modified_df)
        else:
            modified_dfs.append(df)

    mcda_df, mcda_df2 = modified_dfs

    divisor = ((mcda_df['Topsis Score'] != 0).astype(int) +
            (mcda_df2['Topsis Score'] != 0).astype(int))

    #average the TOPSIS scores from the past 3 years
    new_df['Topsis Score'] = (mcda_df['Topsis Score'] + mcda_df2['Topsis Score']) / divisor
    new_df['ticker'] = largest_db['ticker']
    new_df['years'] = f'{year-1}-{year}'
    new_df['tobinsQ'] = (df_og['tobinsQ'] + df2_og['tobinsQ']) / divisor
    
  return average_scores_list


def visualize(average_scores_list):
  for df in average_scores_list:
    df.dropna(inplace=True)
    correlationspear, _ = stats.spearmanr(df['Topsis Score'], df['tobinsQ'])
    
    if not df['years'].empty:
        years = df['years'].iloc[0]
        print(years)
    else:
        print("The 'years' column is empty.")
        years = '2024'
        

    print(f"Spearman's rank correlation for year {years}: {correlationspear} \n")
    sns.set_theme()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Topsis Score', y='tobinsQ', data=df, color='b')
    plt.title(f'Scatter plot of MCDA TOPSIS Score vs Tobin\'s Q for each company, from years {years}')
    plt.xlabel('Topsis Score')
    plt.ylabel('Tobins Q')
    plt.ylim(-0.5, 15)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
  average_scores_list = average_scores()
  visualize(average_scores_list)


