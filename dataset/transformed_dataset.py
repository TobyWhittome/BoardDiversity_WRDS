import pandas as pd
import os


df = pd.read_excel('final_dataset.xlsx')

#Board size mean
df['boardsize_mean'] = df['boardsize'].sub(df['boardsize'].mean()).abs()
del df['boardsize']

excel_file_path = os.path.join(os.getcwd(), 'transformed_dataset.xlsx')
df.to_excel(excel_file_path, index=False)


