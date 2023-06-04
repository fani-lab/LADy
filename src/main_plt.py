import os
import pandas as pd
import openpyxl


def reformatting(address_list):
    for f in address_list:
        output_path = f'{f.replace(".csv", "")}/'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        agg_df = pd.read_csv(f)
        n_aspects = ['5', '10', '15', '20', '25']
        models = ['lda', 'btm', 'octis.neurallda', 'octis.ctm', 'ctm', 'rnd']
        langs_map = {'arb_Arab': 'ar', 'deu_Latn': 'de', 'zho_Hans': 'zh', 'pes_Arab': 'fa', 'spa_Latn': 'es',
                     'fra_Latn': 'fr', 'pes_Arab.zho_Hans.deu_Latn.arb_Arab.fra_Latn.spa_Latn': 'all'}

        new_columns = []
        for col in agg_df.columns:
            if col == 'metric':
                new_columns.append(col)
                continue
            lang = col.split('.')[1]
            if lang in [m.split('.')[0] for m in models]:
                col = col[:col.find('.') + 1] + 'original' + col[col.find('.'):]
                new_columns.append(col)
                continue
            if lang == 'pes_Arab' and 'pes_Arab.zho_Hans.deu_Latn.arb_Arab.fra_Latn.spa_Latn' in col:
                lang = 'pes_Arab.zho_Hans.deu_Latn.arb_Arab.fra_Latn.spa_Latn'
            col = col.replace(lang, langs_map[lang])
            new_columns.append(col)
        agg_df.columns = new_columns

        df_s = {f'{n_a}.{model}': pd.DataFrame() for n_a in n_aspects for model in models}

        for name in df_s:
            for (columnName, columnData) in agg_df.iteritems():
                n_aspect, model_name = name.split('.')[0], name.split('.')[1]
                if f'.{model_name}' in columnName and columnName.startswith(n_aspect):
                    df_s[name] = pd.concat([df_s[name], columnData], axis=1)
            df_s[name].index = [metric.lower() for metric in agg_df.loc[:, 'metric']]

        for name in df_s:
            n_aspect, model_name = name.split('.')[0], name.split('.')[1]
            langs = list(langs_map.values())
            langs.insert(0, 'original')
            df = pd.DataFrame(columns=['0'] + [f'{name}.{i / 10}' for i in range(0, 11)])

            df_langs = {f'{lang}': pd.DataFrame() for lang in langs}

            for lang in df_langs:
                for (columnName, columnData) in df_s[name].iteritems():
                    if lang in columnName:
                        df_langs[lang] = pd.concat([df_langs[lang], columnData], axis=1)
                df_langs[lang].index = df_s[name].index

            for metric in list(df_s[name].index.values):
                row = {col: col for col in df.columns}
                row['0'] = metric
                df.loc[len(df)] = row
                for lang in df_langs:
                    row['0'] = lang
                    for col, val in zip(list(row.keys())[1:], df_langs[lang].loc[metric]):
                        row[col] = val
                    df.loc[len(df)] = row
            df.to_excel(f'{output_path}{name}.xlsx', header=False, index=False)


def plot_graph(input_addresses, show=False):
    import numpy as np
    from matplotlib import pyplot as plt
    import glob
    for input in input_addresses:
        print(f'Generating graphs for {input}...')
        folder = input.replace(".csv", "")
        address_list = glob.glob(folder + "/*.xlsx")

        for address in address_list:
            name = address.replace(folder, "").replace(".xlsx", "")[1:]
            print(f'    Currently at: {name}')
            raw = pd.read_excel(address, header=None)
            n_tbl = int(raw.shape[0] / (raw[raw[0] == 'all'].index[0] + 1))
            df_list = np.array_split(raw, n_tbl)

            for df in df_list:
                marker_list = iter("^x*posPd")
                df.columns = df.iloc[0]
                df = df[1:] # rename column headers
                title = list(df)[0]
                df = df.set_index(list(df)[0]) # rename row indexes

                for labl, row in df.iterrows():
                    plt.xticks(rotation=-45, ha='left')
                    plt.plot(row, label=labl, marker=next(marker_list))

                plt.grid()
                plt.title(title, fontsize=12)
                plt.legend(loc='upper right', bbox_to_anchor=(1.3, 0.99))
                plt.tight_layout()
                plt.savefig(f'{folder}/{folder.replace("../output", "")}.plot/{name}.{title}.pdf', dpi=100, bbox_inches='tight')
                if show: plt.show() # do not show graph by default to save generation time
                plt.close()

if __name__ == '__main__':
    address_list = ['../output/agg.pred.eval.mean-14-latptop.csv',
                    #'../output/agg.pred.eval.mean-14-res.csv',
                    #'../output/agg.pred.eval.mean-15.csv',
                    #'../output/agg.pred.eval.mean-16.csv',
                    ]
    # reformatting(address_list)
    plot_graph(address_list, show=False)
