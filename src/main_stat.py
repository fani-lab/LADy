import pandas as pd
import os

from cmn.review import Review
if __name__ == '__main__':
    datasets = [
        ("C:/Users/tea-n_/Documents/GitHub/LADy/data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml","semeval-16-restaurant"),
        ("C:/Users/tea-n_/Documents/GitHub/LADy/data/raw/semeval/2015SB12/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml", "semeval-15-restaurant"),
        ("C:/Users/tea-n_/Documents/GitHub/LADy/data/raw/semeval/SemEval-14/Laptop_Train_v2.xml","semeval-14-laptop"),
        ("C:/Users/tea-n_/Documents/GitHub/LADy/data/raw/semeval/SemEval-14/Semeval-14-Restaurants_Train.xml", "semeval-14-restaurant"),
        ("C:/Users/tea-n_/Documents/GitHub/LADy/data/raw/twitter/acl-14-short-data/train.raw","twitter"),
        ("C:/Users/tea-n_/Documents/GitHub/LADy/data/raw/mams/train.xml", "mams")
    ]

    dff = pd.DataFrame()
    for data, output in datasets:
        print(data)
        print(output)
        print("==================")
        if "semeval" in data.lower():
            from cmn.semeval import SemEvalReview
            reviews = SemEvalReview.load(data)
        elif "twitter" in data.lower():
            from cmn.twitter import TwitterReview
            reviews = TwitterReview.load(data)
        else:
            from cmn.mams import MAMSReview
            reviews = MAMSReview.load(data)
        
        if not os.path.isdir(output): os.makedirs(output)
        pd.to_pickle(reviews, f'{output}/reviews.pkl')

        stats = Review.get_stats(f'{output}/reviews.pkl', output, plot=False, plot_title=None)
        df = pd.DataFrame.from_dict([stats['*avg_lang_stats']])
        df['nreviews'] = stats['*nreviews']
        df['naspects'] = stats['*naspects']
        df['ntokens'] = stats['*ntokens']
        df['avg_ntokens_review'] = stats['*avg_ntokens_review']
        df['avg_naspects_review'] = stats['*avg_naspects_review']
        df['dataset'] = data
        df.set_index('dataset', inplace=True)
        dff = pd.concat([dff, df])
        print(dff)
        print("==================")
        dff.to_csv(f'{output}stats.csv', index=True)