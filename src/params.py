import random, os, multiprocessing

random.seed(0)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

settings = {
    'cmd': ['prep'],                  # steps of pipeline, ['prep', 'train', 'test', 'eval', 'agg']
    'ncore': multiprocessing.cpu_count(),
    'prep': {
        'doctype': 'snt', # 'rvw' ==> if 'rvw': review = [[review]] else if 'snt': review = [[subreview1], [subreview2], ...]'
        'langaug': ['pes_Arab', 'zho_Hans', 'deu_Latn', 'arb_Arab', 'fra_Latn', 'spa_Latn'],
        # list of nllb language keys to augment via backtranslation from https://github.com/facebookresearch/flores/tree/main/flores200#languages-in-flores-200
        # pes_Arab (Farsi), 'zho_Hans' for Chinese (Simplified), deu_Latn (Germany), spa_Latn (Spanish), arb_Arab (Modern Standard Arabic), fra_Latn (French), ...
        'nllb': 'facebook/nllb-200-distilled-600M',
        'max_l': 400,
        'device': int(os.environ['CUDA_VISIBLE_DEVICES']), #gpu card index
        'batch': False,
        },
    'train': {
        'train_ratio': 0.85, # 1 - train_ratio goes to test
        'nfolds': 5, # on the train, nfold x-valid
        'lda': {'iter_c': 500, 'nwords': 20, 'qualities': ['Coherence', 'Perplexity'],
                'no_extremes': {
                    'no_below': 10,  # happen less than no_below number in total
                    'no_above': 0.9,  # happen in no_above percent of reviews
                    },
                },
        },
    'test': {},
    'eval': {
        'metrics': ['P', 'recall', 'ndcg_cut', 'map_cut', 'success'],
        'topk': '1:1:100',
        'syn': False, #synonyms be added to evaluation
    },
}
