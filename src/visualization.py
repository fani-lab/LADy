import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import params


def plots():
    metrics_list = []
    for m in params.metrics:
        metrics_list.append(f'{m}@k')

    if not os.path.isdir('../output/plots'): os.makedirs(f'../output/plots')

    for naspects in range(5, 55, 5):
        merged = pd.DataFrame()
        metric_idx = 0
        lda = pd.read_csv(f'../output/semeval-2016/txt-version/{naspects}/lda/pred.eval.mean.csv')
        btm = pd.read_csv(f'../output/semeval-2016/txt-version/{naspects}/btm/pred.eval.mean.csv')
        rnd = pd.read_csv(f'../output/semeval-2016/txt-version/{naspects}/rnd/pred.eval.mean.csv')
        merged = pd.concat([lda, btm['mean'], rnd['mean']], axis=1)
        merged.columns = ['Metric', 'LDA', 'BTM', 'RND']
        for i in range(0, 25, 5):
            metric_name = metrics_list[metric_idx]
            query = merged.loc[i:i + 4]
            melted_query = query.melt('Metric', var_name='aspect_models', value_name='Values')
            sns.lineplot(x='Metric', y='Values', hue='aspect_models', palette='Set2', linewidth=3, data=melted_query)
            plt.legend(loc='upper right')  # , title=f"{metric_name.upper()} for {naspects} Topics")
            plt.savefig(f"../output/plots/results_{metric_name}_{naspects}topics.png")
            plt.clf()
            metric_idx += 1


if __name__ == '__main__':
    visualization()

# li = []
# for filename in all_files:
#     df = pd.read_csv(filename, index_col=None, header=0)
#     li.append(df)
# frame = pd.concat(li, axis=0, ignore_index=True)
