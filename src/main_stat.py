import pandas as pd

from cmn.review import Review
if __name__ == '__main__':
    datasets = [
        ('../output/toy.2016SB5', 'toy'),
         ('../output/SemEval-14/Restaurants', 'semeval-14-restaurant'),
          ('../output/SemEval-14/Laptop', 'semeval-14-laptop'),
           ('../output/2015SB12', 'semeval-15-laptop'),
            ('../output/2016SB5', 'semeval-16-laptop'),
            ]
    # dff = pd.DataFrame()
    # for data, title  in datasets:
    #     stats = Review.get_stats(f'{data}/reviews.pes_Arab.zho_Hans.deu_Latn.arb_Arab.fra_Latn.spa_Latn.pkl', data, plot=True, plot_title=None)
    #     df = pd.DataFrame.from_dict([stats['*avg_lang_stats']])
    #     df['nreviews'] = stats['*nreviews']
    #     df['naspects'] = stats['*naspects']
    #     df['ntokens'] = stats['*ntokens']
    #     df['avg_ntokens_review'] = stats['*avg_ntokens_review']
    #     df['avg_naspects_review'] = stats['*avg_naspects_review']
    #     df['dataset'] = data
    #     df.set_index('dataset', inplace=True)
    #     dff = pd.concat([dff, df])
    # dff.to_csv('../output/semeval+/stats.csv', index=True)

    for data, title in datasets:
        Review.plot_semsim_dist(f'{data}/reviews.pes_Arab.zho_Hans.deu_Latn.arb_Arab.fra_Latn.spa_Latn.pkl', f'{data}/reviews.pes_Arab.zho_Hans.deu_Latn.arb_Arab.fra_Latn.spa_Latn.hist.pdf', plot_title=title)


