import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import params
from matplotlib import rcParams


def plots_2d(path, len_topkstr, len_metrics, topic_range):
    metrics_list = []
    for m in params.metrics:
        metrics_list.append(f'{m}@k')

    if not os.path.isdir('../output/plots'): os.makedirs(f'../output/plots')

    for naspects in topic_range:
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


def plots_3d(path, topic_range):
    metrics_dict = {}
    metric_modified_names = {
        "P": "Precision",
        "recall": "Recall",
        "ndcg_cut": "nDCG",
        "map_cut": "MAP",
        "success": "Success",
    }
    k_list = params.topkstr.split(',')
    for m in params.metrics:
        metric_set = []
        for i in k_list:
            metric_set.append(f'{m}_{i}')
        metrics_dict[m] = metric_set

    if not os.path.isdir('../output/plots_3d'): os.makedirs(f'../output/plots_3d')

    merged_lda = merged_btm = merged_rnd = pd.DataFrame()
    naspect_list = []

    for naspects in topic_range:
        naspect_list.append(naspects)
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
        metrics_df_dictionary = {
            "P": model_value[model_value['k'].str.contains("P_")],
            "recall": model_value[model_value['k'].str.contains("recall")],
            "ndcg_cut": model_value[model_value['k'].str.contains("ndcg_cut")],
            "map_cut": model_value[model_value['k'].str.contains("map_cut")],
            "success": model_value[model_value['k'].str.contains("success")]}

        for metric_key, metric_df in metrics_df_dictionary.items():
            fig, ax = plt.subplots(figsize=(10, 10))
            metric_df.reset_index(inplace=True, drop=True)
            metric_df = metric_df.replace(f'{metric_key}_', '', regex=True)
            metric_df = metric_df.astype({"k": int})

            heatmap_pt = pd.pivot_table(metric_df, values='mean', index=['aspects'], columns='k')
            sns.set(font_scale=2)
            h = sns.heatmap(heatmap_pt, cmap='BuPu', xticklabels=True, yticklabels=True, cbar_kws={"shrink": .81})
            h.invert_yaxis()  # change the order of aspects - yaxis
            plt.xticks(rotation=90, fontsize=25)
            plt.yticks(fontsize=25)
            plt.title(f'{metric_modified_names[metric_key]} for {model_key}', fontsize=25, pad=20)
            plt.xlabel('Metrics @K', fontsize=25, labelpad=10)
            plt.ylabel('Number of Aspects', fontsize=25, labelpad=15)

            # hiding labels for clarity that every 5th label is kept
            for ind, label in enumerate(h.get_xticklabels()):
                if (ind + 1) % 10 == 0 or ind == 0:
                    label.set_visible(True)
                else:
                    label.set_visible(False)
            for ind, label in enumerate(h.get_yticklabels()):
                if (ind + 1) % 5 == 0 or ind == 0:
                    label.set_visible(True)
                else:
                    label.set_visible(False)
            ax.set_box_aspect(1)
            plt.savefig(
                f"../output/plots_3d/{path.replace('../output/', '').replace('/', '_')}_{model_key}_{metric_key}.png")
            plt.clf()


def plots_2d_v2(path, len_topkstr, len_metrics, topic_range):
    metrics_list = []
    for m in params.metrics:
        # metrics_list.append(f'{m}@k')
        metrics_list.append(m)
    if not os.path.isdir(f'{path}/new-plots'): os.makedirs(f'{path}/new-plots')

    for naspects in topic_range:
        merged = pd.DataFrame()
        metric_idx = 0

        # topic_model = 'btm'
        # btm_german = pd.read_csv(f'../output/6/xml-2016/{naspects}/{topic_model}/pred.eval.mean.csv')
        # btm_french = pd.read_csv(f'../output/7/xml-2016/{naspects}/{topic_model}/pred.eval.mean.csv')
        # btm_arabic = pd.read_csv(f'../output/8/xml-2016/{naspects}/{topic_model}/pred.eval.mean.csv')
        # btm_chinese = pd.read_csv(f'../output/12/xml-2016/{naspects}/{topic_model}/pred.eval.mean.csv')
        # merged = pd.concat([btm_german, btm_french['mean'], btm_arabic['mean'], btm_chinese['mean']], axis=1)
        # # merged = pd.concat([btm_german, btm_french['mean'], btm_arabic['mean']], axis=1)
        # merged.columns = ['Metric', f'German back-translation using {topic_model}',
        #                   f'French back-translation using {topic_model}',
        #                   f'Arabic back-translation using {topic_model}',
        #                   f'Chinese back-translation using {topic_model}']
        # number_list = len_topkstr
        # metrics_list = pd.DataFrame([], columns=["sentences"])
        btm_before = pd.read_csv(f'../output/13/xml-2016/{naspects}/btm/pred.eval.mean.csv')
        btm_french = pd.read_csv(f'../output/15/xml-2016/{naspects}/btm/pred.eval.mean.csv')
        lda_before = pd.read_csv(f'../output/13/xml-2016/{naspects}/lda/pred.eval.mean.csv')
        lda_french = pd.read_csv(f'../output/15/xml-2016/{naspects}/lda/pred.eval.mean.csv')
        neural_before = pd.read_csv(f'../output/13/xml-2016/{naspects}/neural/pred.eval.mean.csv')
        neural_french = pd.read_csv(f'../output/15/xml-2016/{naspects}/neural/pred.eval.mean.csv')
        rnd_before = pd.read_csv(f'../output/13/xml-2016/{naspects}/rnd/pred.eval.mean.csv')
        rnd_french = pd.read_csv(f'../output/15/xml-2016/{naspects}/rnd/pred.eval.mean.csv')
        merged = pd.concat([btm_before, btm_french['mean'], lda_before['mean'],
                            lda_french['mean'], neural_before['mean'], neural_french['mean'],
                            rnd_before['mean'], rnd_french['mean']], axis=1)
        merged.columns = ['Metric', f'Before back-translation using BTM',
                          f'French back-translation using BTM',
                          f'Before back-translation using LDA',
                          f'French back-translation using LDA',
                          f'Before back-translation using BerTopic',
                          f'French back-translation using BerTopic',
                          f'Before back-translation using Random',
                          f'French back-translation using Random']
        merged['Metric'] = merged['Metric'].str.replace(r'\D', '')
        # before_bt_btm = pd.read_csv(f'../output/9/xml-2016/{naspects}/btm/pred.eval.mean.csv')
        # btm = pd.read_csv(f'../output/10/xml-2016/{naspects}/btm/pred.eval.mean.csv')
        # merged = pd.concat([before_bt_btm, btm['mean']], axis=1)
        # merged.columns = ['Metric', 'Before back-translation BTM', 'German BTM']


        # before_bt_btm = pd.read_csv(f'../output/5/xml-2016/{naspects}/btm/pred.eval.mean.csv')
        # before_bt_lda = pd.read_csv(f'../output/5/xml-2016/{naspects}/lda/pred.eval.mean.csv')
        #
        # btm = pd.read_csv(f'../output/6/xml-2016/{naspects}/btm/pred.eval.mean.csv')
        # lda = pd.read_csv(f'../output/6/xml-2016/{naspects}/lda/pred.eval.mean.csv')
        #
        # merged = pd.concat([before_bt_btm, before_bt_lda['mean'], btm['mean'], lda['mean']], axis=1)
        # merged.columns = ['Metric', 'Before back-translation BTM','Before back-translation LDA', 'German BTM', 'German LDA']


        # before_bt_lda = pd.read_csv(f'../output/5/xml-2016/{naspects}/lda/pred.eval.mean.csv')
        # lda = pd.read_csv(f'../output/6/xml-2016/{naspects}/lda/pred.eval.mean.csv')
        # lda2 = pd.read_csv(f'../output/7/xml-2016/{naspects}/lda/pred.eval.mean.csv')
        # lda3 = pd.read_csv(f'{path}/{naspects}/lda/pred.eval.mean.csv')
        # merged = pd.concat([before_bt_lda, lda['mean'], lda2['mean'], lda3['mean']], axis=1)
        # merged.columns = ['Metric', 'Before back-translation', 'German', 'French', 'Arabic']
        for i in range(0, len_metrics * len_topkstr, len_topkstr):
            metric_name = metrics_list[metric_idx]
            query = merged.loc[i:i + len_topkstr - 1] #len_topkstr
            melted_query = query.melt('Metric', var_name='aspect_models', value_name=metric_name.capitalize())
            h = sns.lineplot(x='Metric', y=metric_name.capitalize(), hue='aspect_models',
                             palette=['Blue', 'Blue', 'Red', 'Red', 'Green', 'Green', 'Orange', 'Orange'],
                             linewidth=3, data=melted_query)
            plt.legend(loc='upper right')

            h.set(xlabel=None)

            leg = plt.legend()
            leg_lines = leg.get_lines()
            leg_lines[0].set_linestyle("solid")
            leg_lines[1].set_linestyle("dashed")
            leg_lines[2].set_linestyle("solid")
            leg_lines[3].set_linestyle("dashed")
            leg_lines[4].set_linestyle("solid")
            leg_lines[5].set_linestyle("dashed")
            leg_lines[6].set_linestyle("solid")
            leg_lines[7].set_linestyle("dashed")

            h.lines[0].set_linestyle("solid")
            h.lines[1].set_linestyle("dashed")
            h.lines[2].set_linestyle("solid")
            h.lines[3].set_linestyle("dashed")
            h.lines[4].set_linestyle("solid")
            h.lines[5].set_linestyle("dashed")
            h.lines[6].set_linestyle("solid")
            h.lines[7].set_linestyle("dashed")

            for ind, label in enumerate(h.get_xticklabels()):
                if (ind + 1) % 20 == 0 or ind == 0:
                    label.set_visible(True)
                else:
                    label.set_visible(False)

            plt.title('Back-translation effect for French language', fontsize=15, pad=20)
            # plt.xlabel('Metrics @K', fontsize=10, labelpad=5)
            plt.ylabel(metric_name.capitalize(), fontsize=10, labelpad=10)

            plt.savefig(
                # f"{path}/new-plots/{path.replace('../output/', '').replace('/', '_')}_{metric_name}_{topic_model}_{naspects}topics.png")
                f"{path}/new-plots/{path.replace('../output/', '').replace('/', '_')}_{metric_name}_baselines_{naspects}topics.png")
            plt.clf()
            metric_idx += 1


def comparison_plot(root_path, eng_path, fra_path, len_topkstr, len_metrics, topic_range):
    metrics_list = params.metrics
    if not os.path.isdir(f'{root_path}/plots'): os.makedirs(f'{root_path}/plots')

    for naspects in topic_range:
        merged = pd.DataFrame()
        metric_idx = 0
        all_before = pd.read_csv(eng_path)
        all_french = pd.read_csv(fra_path)

if __name__ == '__main__':
    # path = ['../output/semeval-2016-full/xml-version', '../output/semeval-2016-full/txt-version']
    # # path = ['../output/semeval-2016-full/xml-version']
    # for p in path:
    #     # plots_2d(p, 100, 5)
    #     topic_range = range(1, 51, 1)
    #     plots_3d(p, topic_range)
    plots_2d_v2('../output/all-French-before-after-synonyms', 100, len(params.metrics), range(25, 30, 5))
    # comparison_plot('../output/English/25', '../output/English/25/25aspects.agg.pred.eval.mean.csv',
    #                 '../output/French/25/25aspects.agg.pred.eval.mean.csv', 100, len(params.metrics), range(25, 30, 5))
