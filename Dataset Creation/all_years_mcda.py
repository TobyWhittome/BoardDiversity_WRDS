import create_dataset
import os
import wrds
import pandas as pd


def output_excel_file(database, filename):
  excel_file_path = os.path.join(os.getcwd(), filename)
  database.to_excel(excel_file_path, index=False)

conn = wrds.Connection(wrds_username="twhittome")

for year in range(2007, 2024):
  
  df = pd.read_excel(f"prev_data/{year}_dataset.xlsx")
  df['Board Size DistFromMean'] = df['boardsize'].sub(df['boardsize'].mean()).abs()
  del df['boardsize']
  num_rows = len(df)
  print(f"Number of rows is {num_rows} for the year {year}")
  output_excel_file(df, f"mcda_prev_data/{year}_dataset.xlsx")
  