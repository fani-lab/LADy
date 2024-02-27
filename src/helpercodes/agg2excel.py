import pandas as pd


def agg(path):
    baselines = ['bert', 'btm', 'cat', 'ctm', 'lda', 'octis.ctm', 'octis.neurallda', 'rnd']
    input_path = f'{path}/agg.ad.pred.eval.mean.csv'
    df = pd.read_csv(input_path)
    filtered_columns = ['metric'] + [col for col in df.columns if col.endswith('0.0')]
    filtered_df = df[filtered_columns]
    filtered_df = filtered_df[filtered_df['metric'].isin(['P_1', 'recall_5', 'ndcg_cut_5', 'map_cut_5'])]

    replacement_dict = {
        'P_1': 'pr1',
        'recall_5': 'rec5',
        'ndcg_cut_5': 'ndcg5',
        'map_cut_5': 'map5'
    }
    filtered_df['metric'] = filtered_df['metric'].map(replacement_dict)

    # specific_names = {'pes_Arab.zho_Hans.deu_Latn.arb_Arab.fra_Latn.spa_Latn': 'all',
    #                   'zho_Hans': 'chinese',
    #                   'pes_Arab': 'farsi',
    #                   'arb_Arab': 'arabic',
    #                   'fra_Latn': 'french',
    #                   'deu_Latn': 'german',
    #                   'spa_Latn': 'spanish'}
    # specific_names = {'fa.zh-CN.de.ar.fr.es': 'all',
    #               'zh-CN': 'chinese',
    #               'fa': 'farsi',
    #               'ar': 'arabic',
    #               'fr': 'french',
    #               'de': 'german',
    #               'es': 'spanish'}
    specific_names = {'lao_Laoo': 'Lao', 'san_Deva': 'Sanskrit'}

    new_columns = []
    for column in filtered_df.columns:
        for substring, replacement in specific_names.items():
            if substring in column:
                column = column.replace(substring, replacement)
                break
        new_columns.append(column)
    filtered_df.columns = new_columns
    filtered_df.columns = [col.replace(".0.0", "").replace("25.", "") for col in filtered_df.columns]

    new_columns = list(filtered_df.columns)
    for i in range(1, len(baselines) + 1):
        new_columns[i] = f'none.{new_columns[i]}'
    # .transpose()
    filtered_df.columns = new_columns
    # filtered_df.to_excel(f'{path}/agg.pred.eval.mean.25.0.0.xlsx', index=False)

    # column_name_parts = ['none', 'chinese', 'farsi', 'arabic', 'french', 'german', 'spanish', 'all']
    column_name_parts = ['Lao', 'Sanskrit']

    transposed_df = pd.DataFrame()
    agg_df = pd.DataFrame()
    agg_list = []
    for part in column_name_parts:
        columns_to_transpose = [col for col in filtered_df.columns if f'{part}.' in col]
        transposed = filtered_df[columns_to_transpose].T
        new_df = transposed.values.reshape(1, -1)
        new_df = pd.DataFrame(new_df,
                              columns=[f'{b}.{m}' for b in baselines for m in replacement_dict.values()])
        new_df.index = [part]
        agg_list.append(new_df)
    agg_df = pd.concat(agg_list)

    agg_df.to_excel(f'{path}/agg.pred.eval.mean.25.0.0.xlsx')


def agg2(path):
    baselines = ['bert', 'btm', 'cat', 'ctm', 'lda', 'octis.ctm', 'octis.neurallda', 'rnd']
    input_path = f'{path}/agg.ad.pred.eval.mean.csv'
    df = pd.read_csv(input_path)
    filtered_columns = ['metric'] + [col for col in df.columns if not col.startswith('metric')]
    filtered_df = df[filtered_columns]
    filtered_df = df[filtered_df['metric'].isin(['P_1', 'ndcg_cut_5'])]

    replacement_dict = {
        'P_1': 'pr1',
        'ndcg_cut_5': 'ndcg5',
    }
    filtered_df['metric'] = filtered_df['metric'].map(replacement_dict)

    specific_names = {'pes_Arab.zho_Hans.deu_Latn.arb_Arab.fra_Latn.spa_Latn': 'all'}

    new_columns = []
    for column in filtered_df.columns:
        for substring, replacement in specific_names.items():
            if substring in column:
                column = column.replace(substring, replacement)
                break
        new_columns.append(column)
    filtered_df.columns = new_columns
    filtered_df.to_excel(f'{path}/agg.pred.eval.mean.25.all.0.0-1.0.xlsx')


if __name__ == '__main__':
    # ['agg-output-low-resource-2015', 'agg-output-low-resource-2016', 'agg-output-low-resource-2014l', 'agg-output-low-resource-2014r']
    for a in ['agg-output-low-resource-2014r']:  # 'agg-output-googletranslate-2015', 'agg-output-googletranslate-2016', 'agg-output-googletranslate-2014r', 'agg-output-googletranslate-2014l'
        agg(a)
        agg2(a)
    # agg('output2')
    # agg('output4-nllb-twitter')
    # agg2('output4-nllb-twitter')
    # agg_00_sheet1('output2/')
    # agg_00_sheet2('output2/')
    # agg_00_sheet3('output2/')
    # agg_0to1_sheet2('output/')
    # agg3('output/')
