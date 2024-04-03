import create_dataset
import os
import wrds

def output_excel_file(database, filename):
  excel_file_path = os.path.join(os.getcwd(), filename)
  database.to_excel(excel_file_path, index=False)

conn = wrds.Connection(wrds_username="twhittome")

for year in range(2007, 2025):
  df = create_dataset.past_data(year, conn)
  
  #Factor & Cluster Analysis
  output_excel_file(df, f"prev_data/{year}_dataset.xlsx")
  
  #MCDA
  df['Board Size DistFromMean'] = df['boardsize'].sub(df['boardsize'].mean()).abs()
  del df['boardsize']
  output_excel_file(df, f"mcda_prev_data/{year}_dataset.xlsx")
  
  

  