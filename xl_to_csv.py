# Code to read, combine, edit slightly, and write excel sheet to csv for faster reading. Run if combined.csv isn't presence in working directory

import pandas as pd

xl_path = r'Final_Four_Experiments_Combined_20240418.xlsx'
dfs = pd.read_excel(xl_path, [1,2])
df = dfs[1].merge(dfs[2])
df.drop(columns=['Gm', 'Drought', 'Nutrient_Deficiency', 'Fs', 'Salinity'], errors='ignore', inplace=True)
df.rename(columns={'Gm.1': 'Gm', 'Drought.1': 'Drought', 'Nutrient_Deficiency.1': 'Nutrient_Deficiency', 'Fs.1': 'Fs', 'Salinity.1': 'Salinity'}, inplace=True)
df.to_csv('combined.csv')