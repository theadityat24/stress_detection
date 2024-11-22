import pandas as pd
import numpy as np
import os

xl_paths = [
    r'1. All data.xlsx',
    r'2. Zeros.xlsx',
    r'3. Mean.xlsx',
    r'4. NAs.xlsx',
]

for path in xl_paths:
    assert(os.path.exists(path))

for i, path in enumerate(xl_paths):
    df = pd.read_excel(path, sheet_name=0)
    stresses = ['Gm', 'Drought', 'Nutrient_Deficiency', 'Fs', 'Salinity']
    df.drop(columns=stresses + ['Soil', 'Fungal_infection'], inplace=True)
    df.rename(columns=dict(zip([s + '.1' for s in stresses], stresses)), inplace=True)

    print(df.shape)
    df.dropna(inplace=True)

    df = df[df['Species'] == 'Black Walnut']
    print(df.shape)

    df.to_csv(f'{i+1}.csv')
