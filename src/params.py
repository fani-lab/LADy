import random, os, multiprocessing
import dotenv

dotenv.load_dotenv()

seed = 0
random.seed(seed)
ncore = multiprocessing.cpu_count()

def to_range(range_str): return range(int(range_str.split(':')[0]), int(range_str.split(':')[2]), int(range_str.split(':')[1]))

settings = {
    'cmd': ['prep', 'train', 'test', 'eval', 'agg'], # steps of pipeline, ['prep', 'train', 'test', 'eval', 'agg']
    'prep': {
        'doctype': 'snt', # 'rvw' # if 'rvw': review => [[review]] else if 'snt': review => [[subreview1], [subreview2], ...]'
        'langaug': ['', 'pes_Arab', 'zho_Hans', 'deu_Latn', 'arb_Arab', 'fra_Latn', 'spa_Latn'], # [''] for no lang augmentation
        # nllb:  ['', 'pes_Arab', 'zho_Hans', 'deu_Latn', 'arb_Arab', 'fra_Latn', 'spa_Latn'] # list of nllb language keys to augment via backtranslation from https://github.com/facebookresearch/flores/tree/main/flores200#languages-in-flores-200 # pes_Arab (Farsi), 'zho_Hans' for Chinese (Simplified), deu_Latn (Germany), spa_Latn (Spanish), arb_Arab (Modern Standard Arabic), fra_Latn (French), ...
        # googletranslate: ['', 'fa', 'zh-CN', 'de', 'ar', 'fr', 'es']
        'translator': 'nllb',  # googletranslate or nllb
        'nllb': 'facebook/nllb-200-distilled-600M',
        'max_l': 1500,
        #https://discuss.pytorch.org/t/using-torch-data-prallel-invalid-device-string/166233
        #gpu card indexes #"cuda:1" if torch.cuda.is_available() else "cpu"
        #cuda:1,2 cannot be used
        'device': f'cuda:{os.environ["CUDA_VISIBLE_DEVICES"]}' if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'cpu',
        'batch': True,
        },
    'train': {
        'for': ['aspect_detection', 'sentiment_analysis'],
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
        'bert': {
            'model_type': 'bert',
            'absa_type': 'tfm',
            'tfm_mode': 'finetune',
            'fix_tfm': 0,
            'model_name_or_path': 'bert-base-uncased',
            'data_dir': '/output/run', # This param will updated dynamically in bert.py
            'task_name': 'lady',
            'per_gpu_train_batch_size': 16,
            'per_gpu_eval_batch_size': 8,
            'learning_rate': 2e-5,
            'do_train': True,
            'do_eval': True,
            'do_lower_case': True,
            'tagging_schema': 'BIEOS',
            'overfit': 0,
            'overwrite_output_dir': True,
            'eval_all_checkpoints': True,
            'MASTER_ADDR': 'localhost',
            'MASTER_PORT': 28512,
            'max_steps': 1200,
            'gradient_accumulation_steps': 1,
            'weight_decay': 0.0,
            'adam_epsilon': 1e-8,
            'max_grad_norm': 1.0,
            'num_train_epochs': 3.0,
            'warmup_steps': 0,
            'logging_steps': 50,
            'save_steps': 100,
            'seed': 42,
            'local_rank': -1,
            'server_ip': '',
            'server_port': '',
            'no_cuda': False,
            'config_name': '',
            'tokenizer_name': '',
            'evaluate_during_training': False,
            'eval_on_testset_after_training': False,
            'no_eval_on_testset_after_training': True,
            'cache_dir': 'bert_cache', # This param will updated dynamically in bert.py

            # test values
            'absa_home': '/output/run/', # This param will updated dynamically in bert.py
            'output_dir': '/output/run/', # This param will updated dynamically in bert.py
            'ckpt': '/checkpoint-1200', # This param will updated dynamically in bert.py
            'max_seq_length': 128,
        },
        'fast': {'epoch': 1000, 'loss': 'ova'}, # ova use independent binary classifiers for each label for multi-label classification
        'lda': {'passes': 1000, 'workers': ncore, 'random_state': seed, 'per_word_topics': True},
        'btm': {'iter': 1000, 'ncore': ncore, 'seed': seed},
        'ctm': {'num_epochs': 1000, 'ncore': ncore, 'seed': seed,
                'bert_model': 'all-mpnet-base-v2',
                'contextual_size': 768,
                'batch_size': 100,
                'num_samples': 10,
                'inference_type': 'combined', #for 'zeroshot' from octis.ctm only
                'verbose': True,
                },
        'octis.neurallda': {'num_epochs': 1000,
                'batch_size': 100,
                'num_samples': 10,
                'ncore': ncore,
                'verbose': True,
                },
        },
    'test': {'h_ratio': 0.0},
    'eval': {
        'for': ['sentiment_analysis', 'aspect_detection'],
        'syn': False, #synonyms be added to evaluation
        'aspect_detection': {
            'metrics': ['P', 'recall', 'ndcg_cut', 'map_cut', 'success'],
            'topkstr': [1, 5, 10, 100], #range(1, 100, 10),
        }, 
        'sentiment_analysis': {
            'metrics': ['recall'],
            'topkstr': [1],
        }
    },
    "dsg": {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "llama_api_key": os.getenv("LLAMA_API_KEY"),
        "output_dir": "./dsg/data/",
        "sys_prompt_identify": "", # Will be used in the future to classify explicit/implicit aspect containing reviews
        "user_prompt_identify": "",
        "sys_prompt_label": """
            You are a data annotator working in the field of review analysis.
            In reviews, customers have a sentiment (positive, neutral, or negative) towards aspects of a product or service.
            You will be given reviews with implicit aspects, where the term is not mentioned explicitly in the text. You are tasked to generate a fitting explicit aspect term.

            You are to output a single word (term) that fits what you judge to be the aspect of the product or service that the review is directing its sentiment towards.
            Do not output any other text aside from the term alone as the system requires the format to be in this output.

            Some examples of potential inputs and outputs:

            Input: Given the review "The quality is not nearly good enough for video calls", in the category laptop, where the sentiment is negative, generate a fitting aspect term.
            Output: webcam

            Input: Given the review "Had to ask him 3 time before he finally corrected my order", in the category restaurant, where the sentiment is negative, generate a fitting aspect term.
            Output: service

            Input: Given the review "It holds a charge for so long", in the category phone, where the sentiment is positive, generate a fitting aspect term.
            Output: battery
        """,
        "user_prompt_label": "Given the review \"%s\", in the category %s, where the sentiment is %s, generate a fitting aspect term.",
        "path": "../data/raw/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml", # Last 3 params altered dynamically at runtime
        "category": "restaurant",
        "model": "gpt-4o-mini",
        "eval": {
            "embedding_model": "text-embedding-3-small", # For evaluating models against ground truth
            "similarity_threshold": 0.5,
        }
    }
}

settings['train']['octis.ctm'] = settings['train']['ctm']
settings['train']['octis.ctm']['inference_type'] = 'zeroshot'
