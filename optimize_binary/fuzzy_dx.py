# python fuzzy_dx.py ..\combined.csv -op fuzzy_dx3.npy.gz -o 3 -w 5

import pandas as pd
import numpy as np
import argparse
import gzip

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='FuzzyDx',
        description='Convolution on spectroscopic data that essentially takes a running-mean discrete derivative. Repeated a given number of times.',
    )

    parser.add_argument('in_path', type=str, help='Should be a csv, where wavelength column names start with "X"')
    parser.add_argument('-op', '--out_path', default=r'fuzzy_dx.npy', type=str, help='Path of final compressed npy.gz file')
    parser.add_argument('-o', '--order', default=1, type=int, help='Number of times to repeat fuzzy dx, only returns last one')
    parser.add_argument('-w', '--window_size', default=5, type=int, help='Size of the filter, sorta like the "fuzziness"')

    args = parser.parse_args()

    csv_path = args.in_path
    df = pd.read_csv(csv_path)
    spec_cols = [col for col in df.columns if col[0] == 'X']

    arr = df[spec_cols].values

    fuzzy_win = args.window_size
    fuzzy_dx_kernel = np.hstack((np.ones(fuzzy_win) * -1/fuzzy_win, np.ones(fuzzy_win)/fuzzy_win))

    def fuzzy_dx(arr):
        return np.apply_along_axis(lambda x: np.convolve(x, fuzzy_dx_kernel, mode='valid'), arr=arr, axis=1)

    for i in range(args.order):
        arr = fuzzy_dx(arr)
        print(f'Calculated order {i+1}')

    f = gzip.GzipFile(args.out_path, "w")
    np.save(file=f, arr=arr)
    f.close()