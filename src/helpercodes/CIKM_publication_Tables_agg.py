import argparse
import pandas as pd
import os
import glob


def agg(output):
    print(f'\nAggregating results in {output} ...')
    files = list()
    for dirpath, dirnames, filenames in os.walk(output):
        files += [os.path.join(os.path.normpath(dirpath), file).split(os.sep) for file in filenames if
                  file.startswith("pred.eval.mean")]

    column_names = []
    path_list = []
    for f in files:
        latency = int(f[2]) / 100.0
        dname = f[1]
        path = ''
        if "-" in dname:
            dname = dname[dname.index("-") + 1:]
            path = f[1][:f[1].index("-")]
        else:
            dname = ''
            path = f[1]
        p = f'{dname}.cat.{latency}'.replace("..", ".")
        if p and p[0] == ".":
            p = p[1:]
        column_names.append(p)
        path_list.append(path)

    column_names.insert(0, 'metric')

    all_results = pd.DataFrame()
    for i, f in enumerate(files):
        df = pd.read_csv(os.sep.join(f))
        if i == 0:
            all_results = df
        else:
            all_results = pd.concat([all_results, df['mean']], axis=1)
    all_results.columns = column_names

    separate_df = {}
    for i, col in enumerate(all_results.columns[1:]):
        name_index = i % len(path_list)
        name = path_list[name_index]
        if name not in separate_df:
            separate_df[name] = pd.DataFrame(all_results.iloc[:, 0])
        separate_df[name][col] = all_results.iloc[:, i + 1]
    for name, df in separate_df.items():
        df.to_csv(f'{output}/{name}_agg.pred.eval.mean.csv', index=False)


def agg_00_sheet1(output):
    print(f'\nAggregating results in {output} for sheet1 ...')
    column_pattern = r"cat\.0\.0"
    file_pattern = f'{output}*_agg.pred.eval.mean.csv'
    file_names = glob.glob(file_pattern)
    df_list = {}
    for file_name in file_names:
        associated_name = file_name.replace(f'{output[:-1]}\\', "").replace("_agg.pred.eval.mean.csv", "")
        df = pd.read_csv(file_name)
        selected_columns = df.filter(regex=column_pattern)
        selected_columns = pd.concat([df.iloc[:, 0], selected_columns], axis=1)
        df_list[associated_name] = pd.DataFrame(selected_columns)

    specific_column_pattern = ['metric', 'cat.0.0', 'arb_Arab.cat.0.0', 'deu_Latn.cat.0.0', 'fra_Latn.cat.0.0',
                               'pes_Arab.cat.0.0',
                               'zho_Hans.cat.0.0', 'spa_Latn.cat.0.0',
                               'pes_Arab.zho_Hans.deu_Latn.arb_Arab.fra_Latn.spa_Latn.cat.0.0']
    for name, df in df_list.items():
        df = df[specific_column_pattern]
        df.to_excel(f'{output}/{name}_sheet1_0.0_agg.pred.eval.mean.xlsx', index=False)


def agg_00_sheet2(output):
    print(f'\nAggregating results in {output} for sheet2 ...')
    file_pattern = f'{output}*_sheet1_0.0_agg.pred.eval.mean.xlsx'
    file_names = glob.glob(file_pattern)
    df_list = {}
    for file_name in file_names:
        associated_name = file_name.replace(f'{output[:-1]}\\', "").replace("_sheet1_0.0_agg.pred.eval.mean.xlsx", "")
        df_list[associated_name] = pd.read_excel(file_name)

    specific_names = {'pes_Arab.zho_Hans.deu_Latn.arb_Arab.fra_Latn.spa_Latn': 'all',
                      'zho_Hans': 'chinese',
                      'pes_Arab': 'farsi',
                      'arb_Arab': 'arabic',
                      'fra_Latn': 'french',
                      'deu_Latn': 'german',
                      'spa_Latn': 'spanish'}
    for name, df in df_list.items():
        new_columns = []
        for column in df.columns:
            for substring, replacement in specific_names.items():
                if substring in column:
                    column = column.replace(substring, replacement)
                    break
            new_columns.append(column)
        df.columns = new_columns
        df.columns = [col.replace(".0.0", "") for col in df.columns]
        df.to_excel(f'{output}/{name}_sheet2_0.0_agg.pred.eval.mean.xlsx', index=False)

        # df = df.transpose()[1:]
        # df.reset_index(drop=True, inplace=True)
        df.transpose().to_excel(f'{output}/{name}_sheet2_transpose_0.0_agg.pred.eval.mean.xlsx', header=False)


def agg_00_sheet3(output):
    print(f'\n Aggregating results in {output} for sheet3 ...')
    file_pattern = f'{output}*_sheet2_transpose_0.0_agg.pred.eval.mean.xlsx'
    file_names = glob.glob(file_pattern)
    df_list = {}
    for file_name in file_names:
        associated_name = file_name.replace(f'{output[:-1]}\\', "").replace(
            "_sheet2_transpose_0.0_agg.pred.eval.mean.xlsx", "")
        df_list[associated_name] = pd.read_excel(file_name)

    for name, df in df_list.items():
        df.iloc[:, 0] = df.iloc[:, 0].str.replace(".cat", "").replace("cat", "none")
        column_filtered_list = ['metric', 'P_1', 'recall_5', 'ndcg_cut_5', 'map_cut_5']
        df = df.loc[:, column_filtered_list]

        lang_order_list = ['none', 'chinese', 'farsi', 'arabic', 'french', 'german', 'spanish', 'all']
        df['metric'] = pd.Categorical(df['metric'], categories=lang_order_list, ordered=True)
        df = df.sort_values('metric')

        df.to_excel(f'{output}/{name}_sheet3_0.0_agg.pred.eval.mean.xlsx', index=False)


def agg_0to1_sheet2(output):
    print(f'\nAggregating results in {output} for sheet2 for 0.0 ro 1.0 latency ...')
    file_pattern = f'{output}*_agg.pred.eval.mean.csv'
    # column_pattern1 = r"pes_Arab.cat.\.0\."
    # column_pattern2 = r"zho_Hans.cat.\.0\."
    # column_pattern3 = r"deu_Latn.cat.\.0\."
    # column_pattern4 = r"fra_Latn.cat.\.0\."
    # column_pattern5 = r"spa_Latn.cat.\.0\."
    # column_pattern6 = r"arb_Arab.cat.\.0\."
    file_names = glob.glob(file_pattern)
    df_list = {}
    for file_name in file_names:
        associated_name = file_name.replace(f'{output[:-1]}\\', "").replace("_agg.pred.eval.mean.csv", "")
        df = pd.read_csv(file_name)
        # selected_columns = df.filter(regex='^(?!.*column_pattern1).*' + '&' + '^(?!.*column_pattern2).*' +
        #                                    '^(?!.*column_pattern3).*' + '&' + '^(?!.*column_pattern4).*' +
        #                                    '&' + '^(?!.*column_pattern5).*' + '&' + '^(?!.*column_pattern6).*')
        # selected_columns = pd.concat([df.iloc[:, ], selected_columns], axis=1)
        # df_list[associated_name] = pd.DataFrame(selected_columns)
        df_list[associated_name] = df

    for name, df in df_list.items():
        df.to_excel(f'{output}/{name}_sheet2_0_to_1_agg.pred.eval.mean.xlsx', index=False)


if __name__ == '__main__':
    agg('output-twitter-modified/')
    agg_00_sheet1('output-twitter-modified/')
    agg_00_sheet2('output-twitter-modified/')
    agg_00_sheet3('output-twitter-modified/')
    # agg_0to1_sheet2('output/')
    # agg3('output/')
