import create_dataset
import os


""" year = 2023
df = create_dataset.main(year)
num_rows = len(df)
print(f"Number of rows is {num_rows} for the year {year}") """

def output_excel_file(database, filename):
  excel_file_path = os.path.join(os.getcwd(), filename)
  database.to_excel(excel_file_path, index=False)

#Currently 2007 is the earliest -- gong to find out which database causes this
# Need to change the S&P500 companies that are being accounted for -- that is why it decreases.

for year in range(2007, 2025):
  df = create_dataset.main(year)
  num_rows = len(df)
  print(f"Number of rows is {num_rows} for the year {year}")
  output_excel_file(df, f"prev_data/{year}_dataset.xlsx")
  