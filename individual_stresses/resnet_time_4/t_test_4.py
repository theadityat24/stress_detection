import numpy as np
import pandas as pd
import scipy

results_path = r'k_folds_results.csv'

df = pd.read_csv(results_path)

paths = df['path'].unique()

# test differences between datasets

results = {'path1': [], 'path2': [], 'acc_t': [], 'c_acc_t': [], 'acc_p': [], 'c_acc_p': [],}

for i, path1 in enumerate(paths[:-1]):
    for j, path2 in enumerate(paths[i+1:]):
        results['path1'].append(path1)
        results['path2'].append(path2)

        t, p = scipy.stats.ttest_ind(
            df[df['path']==path1]['acc'], df[df['path']==path2]['acc'], equal_var=False
        )
        results['acc_t'].append(t)
        results['acc_p'].append(p)

        c_t, c_p = scipy.stats.ttest_ind(
            df[df['path']==path1]['c_acc'], df[df['path']==path2]['c_acc'], equal_var=False
        )
        results['c_acc_t'].append(c_t)
        results['c_acc_p'].append(c_p)

results_df = pd.DataFrame(results)
print(results_df)

# test differences between acc and c_acc
results2 = {'path': [], 't': [], 'p': []}
for path in paths:
    results2['path'].append(path)
    t, p = scipy.stats.ttest_rel(
        df[df['path']==path]['acc'], df[df['path']==path2]['c_acc']
    )
    results2['t'].append(t)
    results2['p'].append(p)

results2_df = pd.DataFrame(results2)
print(results2_df)