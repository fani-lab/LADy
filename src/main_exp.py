# #################################################################
# # experiments ...
# # to run pipeline for datasets * baselines * naspects * hide_ratios
import argparse

import params
import main
import main_octis

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latent Aspect Detection')
    parser.add_argument('-am', type=str.lower, default='rnd', help='aspect modeling method (eg. --am lda)')
    parser.add_argument('-data', dest='data', type=str, help='raw dataset file path, e.g., -data ..data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml')
    parser.add_argument('-output', dest='output', type=str, default='../output/', help='output path, e.g., -output ../output/semeval/2016.xml')
    parser.add_argument('-naspects', dest='naspects', type=int, default=25, help='user-defined number of aspects, e.g., -naspect 25')
    args = parser.parse_args()

    datasets = [('../data/raw/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml', '../output/semeval+/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml'),
                # ('../data/raw/semeval/SemEval-14/Semeval-14-Restaurants_Train.xml', '../output/semeval+/SemEval-14/Semeval-14-Restaurants_Train.xml'),
                # ('../data/raw/semeval/2015SB12/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml', '../output/semeval+/2015SB12/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml'),
                # ('../data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml', '../output/semeval+/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml')
                ]

    octis = True
    for (data, output) in datasets:
        args.data = data
        args.output = output
        params.settings['prep']['langaug'] = ['', 'pes_Arab', 'zho_Hans', 'deu_Latn', 'arb_Arab', 'fra_Latn', 'spa_Latn']
        params.settings['cmd'] = ['prep']
        main_octis.main(args) if octis else main.main(args)
        langs = params.settings['prep']['langaug'].copy()
        langs.extend([params.settings['prep']['langaug']])
        for lang in langs:
            params.settings['prep']['langaug'] = lang if isinstance(lang, list) else [lang]
            for am in ['ctm', 'nrl'] if octis else ['rnd', 'lda', 'btm', 'ctm',]:
                for naspects in [5]:#range(5, 30, 5):
                    for hide in range(0, 110, 10):
                        args.am = am
                        args.naspects = naspects
                        # # to train on entire dataset only
                        # params.settings['train']['ratio'] = 0.999
                        # params.settings['train']['nfolds'] = 0
                        params.settings['test']['h_ratio'] = round(hide * 0.01, 1)
                        params.settings['cmd'] = ['prep', 'train', 'test']
                        main_octis.main(args) if octis else main.main(args)
            if 'agg' in params.settings['cmd']: main.agg(args.output, args.output)