import os
import pandas as pd


def agg(path, output):
    print(f'\n5. Aggregating results in {path} in {output} ...')

    files = list()

    for dirpath, _, filenames in os.walk(path):
        files += [
            os.path.join(os.path.normpath(dirpath), file).split(os.sep)
            for file in filenames
            if file.startswith(f'model.ad.pred.eval.mean')
            ]

    column_names = []
    for f in files:
        p = '.'.join(f[-3:]).replace('.csv', '').replace(f'model.ad.pred.eval.mean.', '')
        column_names.append(p)
    column_names.insert(0, 'metric')

    all_results = pd.DataFrame()
    for i, f in enumerate(files):
        df = pd.read_csv(os.sep.join(f))
        if i == 0: all_results = df
        else: all_results = pd.concat([all_results, df['mean']], axis=1)

    all_results.columns = column_names
    all_results.to_csv(f'{output}/agg.ad.pred.eval.mean.csv', index=False)


if __name__ == '__main__':
    for a in ['twitter']:
        agg(a, a)
