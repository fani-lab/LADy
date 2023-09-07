# #################################################################
# # experiments ...
# # to run pipeline for datasets * baselines * naspects * hide_ratios
import argparse

import params
import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latent Aspect Detection')
    parser.add_argument('-am', type=str.lower, default='rnd', help='aspect modeling method (eg. --am lda)')
    parser.add_argument('-data', dest='data', type=str, help='raw dataset file path, e.g., -data ..data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml')
    parser.add_argument('-output', dest='output', type=str, default='../output/', help='output path, e.g., -output ../output/semeval/2016.xml')
    parser.add_argument('-naspects', dest='naspects', type=int, default=25, help='user-defined number of aspects, e.g., -naspect 25')
    args = parser.parse_args()

    datasets = [
                ('../data/raw/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml', '../output/toy.2016SB5-new'),
                ]

    octis = True
    for (data, output) in datasets:
        args.data = data
        args.output = output
        params.settings['prep']['langaug'] = ['fra_Latn', 'spa_Latn']
        params.settings['cmd'] = ['prep']
        params.settings['train']['fold'] = 2
        params.settings['eval']['metrics'] = ['P', 'ndcg_cut']
        params.settings['eval']['topk'] = [1, 10]

        # main_octis.main(args) if octis else main.main(args)
        main.main(args)
        langs = params.settings['prep']['langaug'].copy()
        langs.extend([params.settings['prep']['langaug']])
        for lang in langs:
            params.settings['prep']['langaug'] = lang if isinstance(lang, list) else [lang]
            for am in ['rnd', 'lda', 'btm', 'ctm', 'octis.neurallda', 'octis.ctm']:
                for naspects in range(5, 30, 20):
                    for hide in range(0, 110, 50):
                        args.am = am
                        args.naspects = naspects
                        # # to train on entire dataset only
                        # params.settings['train']['ratio'] = 0.999
                        # params.settings['train']['nfolds'] = 0
                        params.settings['test']['h_ratio'] = round(hide * 0.01, 1)
                        params.settings['cmd'] = ['agg']
                        # main_octis.main(args) if octis else main.main(args)
                        # main.main(args)
        if 'agg' in params.settings['cmd']: main.agg(args.output, args.output)