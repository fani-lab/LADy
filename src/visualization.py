import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import params


def plots_2d(path, len_topkstr, len_metrics):
    metrics_list = []
    for m in params.metrics:
        metrics_list.append(f'{m}@k')

    if not os.path.isdir('../output/plots'): os.makedirs(f'../output/plots')

    for naspects in range(5, 55, 5):
        merged = pd.DataFrame()
        metric_idx = 0
        lda = pd.read_csv(f'{path}/{naspects}/lda/pred.eval.mean.csv')
        btm = pd.read_csv(f'{path}/{naspects}/btm/pred.eval.mean.csv')
        rnd = pd.read_csv(f'{path}/{naspects}/rnd/pred.eval.mean.csv')
        merged = pd.concat([lda, btm['mean'], rnd['mean']], axis=1)
        merged.columns = ['Metric', 'LDA', 'BTM', 'RND']
        for i in range(0, len_metrics * len_metrics, len_topkstr):
            metric_name = metrics_list[metric_idx]
            query = merged.loc[i:i + len_topkstr - 1]
            melted_query = query.melt('Metric', var_name='aspect_models', value_name='Values')
            sns.lineplot(x='Metric', y='Values', hue='aspect_models', palette='Set2', linewidth=3, data=melted_query)
            plt.legend(loc='upper right')  # , title=f"{metric_name.upper()} for {naspects} Topics")
            plt.savefig(
                f"../output/plots/{path.replace('../output/', '').replace('/', '_')}_{metric_name}_{naspects}topics.png")
            plt.clf()
            metric_idx += 1


def plots_3d(path):
    if not os.path.isdir('../output/plots_3d'): os.makedirs(f'../output/plots_3d')

    merged_lda = pd.DataFrame()
    merged_btm = pd.DataFrame()
    merged_rnd = pd.DataFrame()
    for naspects in range(5, 55, 5):
        lda = pd.read_csv(f'{path}/{naspects}/lda/pred.eval.mean.csv')
        btm = pd.read_csv(f'{path}/{naspects}/btm/pred.eval.mean.csv')
        rnd = pd.read_csv(f'{path}/{naspects}/rnd/pred.eval.mean.csv')

        lda['aspects'] = naspects
        btm['aspects'] = naspects
        rnd['aspects'] = naspects

        merged_lda = pd.concat([merged_lda, lda], axis=0)
        merged_btm = pd.concat([merged_btm, btm], axis=0)
        merged_rnd = pd.concat([merged_rnd, rnd], axis=0)

    merged_lda.columns = merged_btm.columns = merged_rnd.columns = ['k', 'mean', 'aspects']

    models = {"LDA": merged_lda, "BTM": merged_btm, "RND": merged_rnd}

    for model_key, model_value in models.items():
        fig, ax = plt.subplots(figsize=(10, 10))
        metrics_df_dictionary = {
            "Precision": model_value[model_value['k'].str.contains("P_")],
            "Recall": model_value[model_value['k'].str.contains("recall")],
            "Ndcg cut": model_value[model_value['k'].str.contains("ndcg_cut")],
            "Map cut": model_value[model_value['k'].str.contains("map_cut")],
            "Success": model_value[model_value['k'].str.contains("success")]}
        for metric_key, metric_value in metrics_df_dictionary.items():
            data = metric_value[['k', 'mean', 'aspects']]

            # y = data['k'].unique().tolist()
            # order = []
            # for index, row in data.iterrows():
            #     order.append(y.index(row['k']))
            # data['order'] = order
            # data = data.sort_values(by=['order'])
            # data = data.reset_index(drop=True)
            # data = data.drop('order', axis=1)

            heatmap_pt = pd.pivot_table(data, values='mean', index=['aspects'], columns='k')
            sns.set()
            sns.heatmap(heatmap_pt, cmap='BuPu')
            plt.xticks(rotation=15)
            plt.title(f'{metric_key} for {model_key}', fontsize=30, pad=30)
            plt.xlabel('Metrics', fontsize=20, labelpad=20)
            plt.ylabel('Number of Aspects', fontsize=20, labelpad=20)
            plt.savefig(
                f"../output/plots_3d/{path.replace('../output/', '').replace('/', '_')}_{model_key}_{metric_key}.png")
            plt.clf()


if __name__ == '__main__':
    path = ['../output/semeval-2016/xml-version', '../output/semeval-2016/txt-version']
    for p in path:
        # plots_2d(p, 100, 5)
        plots_3d(p)
