import pandas as pd

from cmn.review import Review
if __name__ == '__main__':
    datasets = ['../output/semeval+/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml',
                '../output/semeval+/SemEval-14/Semeval-14-Restaurants_Train.xml',
                '../output/semeval+/SemEval-14/Laptop_Train_v2.xml',
                '../output/semeval+/2015SB12/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml',
                '../output/semeval+/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml'
            ]
    dff = pd.DataFrame()
    for data in datasets:
        stats = Review.get_stats(f'{data}/reviews.pes_Arab.zho_Hans.deu_Latn.arb_Arab.fra_Latn.spa_Latn.pkl', data, plot=False, plot_title=None)
        df = pd.DataFrame.from_dict([stats['*avg_lang_stats']])
        df['nreviews'] = stats['*nreviews']
        df['naspects'] = stats['*naspects']
        df['ntokens'] = stats['*ntokens']
        df['avg_ntokens_review'] = stats['*avg_ntokens_review']
        df['avg_naspects_review'] = stats['*avg_naspects_review']
        df['dataset'] = data
        df.set_index('dataset', inplace=True)
        dff = pd.concat([dff, df])
    dff.to_csv('../output/semeval+/stats.csv', index=True)

