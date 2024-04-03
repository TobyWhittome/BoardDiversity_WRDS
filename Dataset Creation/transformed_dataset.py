import pandas as pd
import os


df = pd.read_excel('dataset/final_dataset.xlsx')

#Board size mean
df['Board Size DistFromMean'] = df['Board Size'].sub(df['Board Size'].mean()).abs()
del df['Board Size']

excel_file_path = os.path.join(os.getcwd(), 'dataset/transformed_dataset.xlsx')
df.to_excel(excel_file_path, index=False)


