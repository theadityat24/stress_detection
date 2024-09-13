# Code to read, combine, edit slightly, and write excel sheet to csv for faster reading. Run if combined.csv isn't present in working directory

import pandas as pd

xl_path = r'Final_Four_Experiments_Combined_20240418.xlsx'
dfs = pd.read_excel(xl_path, [1,2])

stresses = ['Gm', 'Drought', 'Nutrient_Deficiency', 'Fs', 'Salinity']

for i in dfs.keys():
    dfs[i].drop(columns=stresses, errors='ignore', inplace=True)
    dfs[i].rename(columns=dict(zip([s + '.1' for s in stresses], stresses)), inplace=True)

df = dfs[1].join(dfs[2], how='left', lsuffix='_')
df.drop(columns=[s for s in df.columns if (s[-1] == '_')], inplace=True)
df.to_csv('combined.csv')