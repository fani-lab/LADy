import random, os, multiprocessing

seed = 0
random.seed(seed)
ncore = multiprocessing.cpu_count()

def to_range(range_str): return range(int(range_str.split(':')[0]), int(range_str.split(':')[2]), int(range_str.split(':')[1]))

settings = {
    'cmd': ['prep', 'train', 'test'], # steps of pipeline, ['prep', 'train', 'test', 'eval', 'agg']
    'prep': {
        'doctype': 'snt', # 'rvw' # if 'rvw': review => [[review]] else if 'snt': review => [[subreview1], [subreview2], ...]'
        'langaug': [''],#, 'pes_Arab', 'zho_Hans', 'deu_Latn', 'arb_Arab', 'fra_Latn', 'spa_Latn'], #[''] for no lang augmentation
        # list of nllb language keys to augment via backtranslation from https://github.com/facebookresearch/flores/tree/main/flores200#languages-in-flores-200
        # pes_Arab (Farsi), 'zho_Hans' for Chinese (Simplified), deu_Latn (Germany), spa_Latn (Spanish), arb_Arab (Modern Standard Arabic), fra_Latn (French), ...
        'nllb': 'facebook/nllb-200-distilled-600M',
        'max_l': 1024,
        #https://discuss.pytorch.org/t/using-torch-data-prallel-invalid-device-string/166233
        #gpu card indexes #"cuda:1" if torch.cuda.is_available() else "cpu"
        #cuda:1,2 cannot be used
        'device': f'cuda:{os.environ["CUDA_VISIBLE_DEVICES"]}' if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'cpu',
        'batch': True,
        },
    'train': {
        'ratio': 0.85, # 1 - ratio goes to test. To train on entire dataset: 0.999 and 'nfolds': 0
        'nfolds': 5, # on the train, nfold x-valid, 0: no x-valid only test and train, 1: test, 1-fold
        'langaug_semsim': 0.5, # backtranslated review is in training if its semantic similarity with original review is >= this value
        'nwords': 20,
        'qualities': ['coherence', 'perplexity'],
        'quality': ['u_mass'],#[, 'c_v', 'c_uci', 'c_npmi'],
        'no_extremes': None,
                # {'no_below': 10,   # happen less than no_below number in total
                #  'no_above': 0.9}  # happen in no_above percent of reviews
        'rnd': {},
        'lda': {'passes': 1000, 'workers': ncore, 'random_state': seed, 'per_word_topics': True},
        'btm': {'iter': 1000, 'ncore': ncore, 'seed': seed},
        'ctm': {'num_epochs': 1000, 'ncore': ncore, 'seed': seed,
                'bert_model': 'all-mpnet-base-v2',
                'contextual_size': 768,
                'batch_size': 100,
                'num_samples': 10,
                'inference_type': 'combined', #for 'zeroshot' from octis.ctm only
                'ncore': ncore,
                'verbose': True,
                },
        'octis.neurallda': {'num_epochs': 1000,
                'batch_size': 100,
                'num_samples': 10,
                'ncore': ncore,
                'verbose': True,
                },
        },
    'test': {'h_ratio': 1.0},
    'eval': {
        'metrics': ['P', 'recall', 'ndcg_cut', 'map_cut', 'success'],
        'topkstr': [1, 5, 10, 100], #range(1, 100, 10),
        'syn': False, #synonyms be added to evaluation
    },
}

settings['train']['octis.ctm'] = settings['train']['ctm']
settings['train']['octis.ctm']['inference_type'] = 'zeroshot'
