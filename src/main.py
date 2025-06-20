from typing import List, Tuple, Set, Union
import argparse, os, json, time
from tqdm import tqdm
import numpy as np, pandas as pd
import pampy

import hydra
from omegaconf import DictConfig

import pytrec_eval
from nltk.corpus import wordnet as wn

import params
from cmn.review import Review
## As requirement.txt is not installing these models for now, therefore this line is commented out
from aml.mdl import AbstractAspectModel, AbstractSentimentModel, ModelCapabilities, ModelCapability # 
# ---------------------------------------------------------------------------------------
# Typings
# ---------------------------------------------------------------------------------------
Aspects = List[List[str]]

PredictedAspect = List[Tuple[int, float]]  # Tuple containing index and weight

PairType = Tuple[Aspects, PredictedAspect]

# ---------------------------------------------------------------------------------------
# Logics
# ---------------------------------------------------------------------------------------
def load(input, output, cfg, cache=True):
    print('\n1. Loading reviews and preprocessing ...')
    print('#' * 50)
    try:
        if not cache: raise FileNotFoundError
        print(f'1.1. Loading existing processed reviews file {output}...')
        return pd.read_pickle(output)

    except (FileNotFoundError, EOFError) as _:
        try:
            print('1.1. Loading existing processed pickle file failed! Loading raw reviews ...')
            if "semeval" in input.lower():
                from cmn.semeval import SemEvalReview
                reviews = SemEvalReview.load(input, explicit= True, implicit=True)
            elif "twitter" in input.lower():
                from cmn.twitter import TwitterReview
                reviews = TwitterReview.load(input)
            elif "mams" in input.lower():
                from cmn.mams import MAMSReview
                reviews = MAMSReview.load(input)
            else:
                # from cmn.mams import MAMSReview
                print("No specific dataset ('semeval' or 'twitter' or 'mams') was detected in the input.")
            print(f'(#reviews: {len(reviews)})')
            if(len(reviews) != 0):
                print(f'\n1.2. Augmentation via backtranslation by {cfg.prep.langaug} {"in batches" if params.settings["prep"] else ""}...')
                for lang in cfg.prep.langaug:
                    if lang:
                        print(f'\n{lang} ...')
                        if cfg.prep.batch:
                            start = time.time()
                            Review.translate_batch(reviews, lang, cfg.prep) #all at once, esp., when using gpu
                            end = time.time()
                            print(f'{lang} done all at once (batch). Time: {end - start}')
                        else:
                            for r in tqdm(reviews): r.translate(lang, cfg.prep)

                    # to save a file per langauge. I know, it has a minor logical bug as the save file include more langauges!
                    output_ = output
                    for l in cfg.prep.langaug:
                        if l and l != lang:
                            output_ = output_.replace(f'{l}.', '')
                    pd.to_pickle(reviews, output_)

                print(f'\n1.3. Saving processed pickle file {output}...')
                pd.to_pickle(reviews, output)
                return reviews
        except Exception as error:
            print(f'Error...{error}')
            raise error

def split(nsample, output):
    # We split originals into train, valid, test. So each have its own augmented versions.
    # During test (or even train), we can decide to consider augmented version or not.

    from sklearn.model_selection import KFold, train_test_split
    from json import JSONEncoder

    train, test = train_test_split(np.arange(nsample), train_size=params.settings['train']['ratio'], random_state=params.seed, shuffle=True)

    splits = dict()
    splits['test'] = test
    splits['folds'] = dict()
    if params.settings['train']['nfolds'] == 0:
        splits['folds'][0] = dict()
        splits['folds'][0]['train'] = train
        splits['folds'][0]['valid'] = []
    elif params.settings['train']['nfolds'] == 1:
        splits['folds'][0] = dict()
        splits['folds'][0]['train'] = train[:len(train)//2]
        splits['folds'][0]['valid'] = train[len(train)//2:]
    else:
        skf = KFold(n_splits=params.settings['train']['nfolds'], random_state=params.seed, shuffle=True)
        for k, (trainIdx, validIdx) in enumerate(skf.split(train)):
            splits['folds'][k] = dict()
            splits['folds'][k]['train'] = train[trainIdx]
            splits['folds'][k]['valid'] = train[validIdx]

    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            return JSONEncoder.default(self, obj)

    with open(f'{output}/splits.json', 'w') as f: json.dump(splits, f, cls=NumpyArrayEncoder, indent=1)
    return splits

def train(am, train, valid, f, output, capability: ModelCapability, cfg):
    print(f'\n2. Aspect model training for {am.name()} ...')
    print('#' * 50)
    try:
        print(f'2.1. Loading saved aspect model from {output}/f{f}. ...')
        am.load(f'{output}/f{f}.')
    except (FileNotFoundError, EOFError) as _:
        print(f'2.1. Loading saved aspect model failed! Training {am.name()} for {cfg.args.naspects} of aspects. See {output}/f{f}.model.train.log for training logs ...')
        if not os.path.isdir(output): os.makedirs(output)
        get_model_train_method(am, capability)(train, valid, params.settings['train'][am.name()], params.settings['prep']['doctype'], params.settings['train']['no_extremes'], f'{output}/f{f}.')

        # from aml.mdl import AbstractAspectModel
        print('2.2. Quality of aspects ...')
        for q in params.settings['train']['qualities']: print(f'({q}: {am.quality(q)})')

def test(am, test, f, output: str, capability: ModelCapability):
    cp_name = get_capability_short_name(capability)
    model_job = capability.split("_")[0].capitalize()

    print(f'\n3. {model_job} model testing for {am.name()} ...')
    print('#' * 50)
    try:
        print(f'\n3.1. Loading saved predictions on test set from {output}f{f}.{cp_name}.model.pred.{params.settings["test"]["h_ratio"]} ...')
        return pd.read_pickle(f'{output}f{f}.model.{cp_name}.pred.{params.settings["test"]["h_ratio"]}')
    except (FileNotFoundError, EOFError) as _:
        print(f'\n3.1. Loading saved predictions on test set failed! Predicting on the test set with {params.settings["test"]["h_ratio"] * 100}% latent aspect ...')
        print(f'3.2. Loading {model_job} model from {output}f{f}.{cp_name}.model for testing ...')
        am.load(f'{output}/f{f}.')
        print(f'3.3. Testing {model_job} model ...')
        pairs = get_model_infer_method(am, capability)(reviews_test=test, h_ratio=params.settings['test']['h_ratio'], doctype=params.settings['prep']['doctype'], output=f'{output}/f{f}')
        pd.to_pickle(pairs, f'{output}f{f}.model.{cp_name}.pred.{params.settings["test"]["h_ratio"]}')

def get_model_infer_method(am: Union[AbstractSentimentModel, AbstractAspectModel], model_capability: ModelCapability):
    if isinstance(am, AbstractAspectModel) and model_capability == 'aspect_detection':
        return am.infer_batch
    elif isinstance(am, AbstractSentimentModel) and model_capability == 'sentiment_analysis':
        return am.infer_batch_sentiment
    
    raise Exception(f'Not handled model: {am.name()}')

def get_model_train_method(am: Union[AbstractSentimentModel, AbstractAspectModel], model_capability: ModelCapability):
    if isinstance(am, AbstractAspectModel) and model_capability == 'aspect_detection':
        return am.train
    elif isinstance(am, AbstractSentimentModel) and model_capability == 'sentiment_analysis':
        return am.train_sentiment
    
    raise Exception(f'Not handled model: {am.name()}')


def get_model_metrics(model_capability: ModelCapability) -> Set[str]:
    return pampy.match(model_capability, 
        'aspect_detection', set(f'{m}_{",".join([str(i) for i in params.settings["eval"]["aspect_detection"]["topkstr"]])}' for m in params.settings['eval']['aspect_detection']['metrics']),
        'sentiment_analysis', set(f'{m}_{",".join([str(i) for i in params.settings["eval"]["sentiment_analysis"]["topkstr"]])}' for m in params.settings['eval']['sentiment_analysis']['metrics']),
    ) #type: ignore

def get_capability_short_name(cp: ModelCapability) -> str:
    return pampy.match(cp,
        'sentiment_analysis', 'sa',
        'aspect_detection', 'ad',
    ) # type: ignore

def evaluate(input: str, output: str, model_capability: ModelCapability):
    model_job = model_capability.split('_')[0].capitalize()
    print(f'\n4. {model_job} model evaluation for {input} ...')
    print('#' * 50)
    pairs = pd.read_pickle(input)
    metrics_set = get_model_metrics(model_capability)

    qrel = dict()
    run = dict()

    print(f'\n4.1. Building pytrec_eval input for {len(pairs)} instances ...')
    for i, pair in enumerate(pairs):
        if params.settings['eval']['syn']:
            syn_list = set()
            for p_instance in pair[0]:
                syn_list.add(p_instance)
                syn_list.update(set([lemma.name() for syn in wn.synsets(p_instance) for lemma in syn.lemmas()]))
            qrel['q' + str(i)] = {w: 1 for w in syn_list}
        else:
            qrel['q' + str(i)] = {w: 1 for w in pair[0]}

        # the prediction list may have duplicates
        run['q' + str(i)] = {}
        for j, (w, p) in enumerate(pair[1]):
            if w not in run['q' + str(i)].keys():
                run['q' + str(i)][w] = len(pair[1]) - j

    print(f'4.2. Calling pytrec_eval for {metrics_set} ...')
    df = pd.DataFrame.from_dict(pytrec_eval.RelevanceEvaluator(qrel, metrics_set).evaluate(run))  # qrel should not have empty entry otherwise get exception
    print('4.3. Averaging ...')
    df_mean = df.mean(axis=1).to_frame('mean')
    df_mean.to_csv(output)
    return df_mean

def agg(path, output):
    capabilites: ModelCapabilities = ['aspect_detection', 'sentiment_analysis']
    cp_short_names = list(map(get_capability_short_name, capabilites))

    print(f'\n5. Aggregating results in {path} in {output} ...')

    for cp in cp_short_names:
        files = list()

        for dirpath, _, filenames in os.walk(path):
            files += [
                os.path.join(os.path.normpath(dirpath), file).split(os.sep)
                for file in filenames
                if file.startswith(f'model.{cp}.pred.eval.mean')
                ]

        column_names = []
        for f in files:
            p = '.'.join(f[-3:]).replace('.csv', '').replace(f'model.{cp}.pred.eval.mean.', '')
            column_names.append(p)
        column_names.insert(0, 'metric')

        all_results = pd.DataFrame()
        for i, f in enumerate(files):
            df = pd.read_csv(os.sep.join(f))
            if i == 0: all_results = df
            else: all_results = pd.concat([all_results, df['mean']], axis=1)

        all_results.columns = column_names
        all_results.to_csv(f'{output}/agg.{cp}.pred.eval.mean.csv', index=False)
        
def parse_args():
    parser = argparse.ArgumentParser(description='Latent Aspect Detection')
    parser.add_argument('-am', type=str.lower, default='rnd', help='aspect modeling method (eg. --am lda)')
    parser.add_argument('-data', dest='data', type=str, default='./data/raw/mams/test.xml', help='raw dataset file path, e.g., -data ..data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml')
    parser.add_argument('-output', dest='output', type=str, default='./output/mams-agg', help='output path, e.g., -output ../output/semeval/2016.xml')
    parser.add_argument('-naspects', dest='naspects', type=int, default=25, help='user-defined number of aspects, e.g., -naspect 25')
    return parser.parse_args()

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # args = parse_args()
    if 'prep' in cfg.cmd:
        if not os.path.isdir(cfg.args.output): os.makedirs(cfg.args.output)
        langaug_str = '.'.join([l for l in cfg.prep.langaug if l])
        reviews = load(cfg.args.data, f'{cfg.args.output}/reviews.{langaug_str}.pkl'.replace('..pkl', '.pkl'), cfg)
        if(reviews != None):
            if(len(reviews) > 2):
                splits = split(len(reviews), cfg.args.output)
        output = f'{cfg.args.output}/{cfg.args.naspects}.{langaug_str}'.rstrip('.')
    
    if 'llmargs' in cfg.cmd:
        from llm.aspect_extraction_pipeline import mainllm
        reviews = mainllm(cfg, reviews)

    if any(x in cfg.cmd for x in ['train', 'test', 'eval']):
        am = None

        if not os.path.isdir(output): os.makedirs(output)
        if 'rnd' == cfg.args.am: from aml.rnd import Rnd; am = Rnd(cfg.args.naspects, params.settings['train']['nwords']) 
        if 'lda' == cfg.args.am: from aml.lda import Lda; am = Lda(cfg.args.naspects, params.settings['train']['nwords'])
        if 'btm' == cfg.args.am: from aml.btm import Btm; am = Btm(cfg.args.naspects, params.settings['train']['nwords'])
        if 'ctm' == cfg.args.am: from aml.ctm import Ctm; am = Ctm(cfg.args.naspects, params.settings['train']['nwords'], params.settings['train']['ctm']['contextual_size'], params.settings['train']['ctm']['num_samples'])
        if 'bert' == cfg.args.am: from aml.bert import BERT; am = BERT(cfg.args.naspects, params.settings['train']['nwords'])
        if 'fast' == cfg.args.am: from aml.fast import Fast; am = Fast(cfg.args.naspects, params.settings['train']['nwords'])
        if 'octis.ctm' == cfg.args.am: from octis.models.CTM import CTM; from aml.nrl import Nrl; am = Nrl(CTM(), cfg.args.naspects, params.settings['train']['nwords'], params.settings['train']['quality'])
        if 'octis.neurallda' == cfg.args.am: from octis.models.NeuralLDA import NeuralLDA; from aml.nrl import Nrl; am = Nrl(NeuralLDA(), cfg.args.naspects, params.settings['train']['nwords'], params.settings['train']['quality'])


        if(am is None): raise Exception('Model not found!')

        output = f'{output}/{am.name()}/'

        eval_for: ModelCapabilities = set(params.settings['eval']['for']).intersection(am.capabilities) #type: ignore
        train_for: ModelCapabilities = set(params.settings['train']['for']).intersection(am.capabilities) #type: ignore

        if 'train' in cfg.cmd:
            for capability in train_for:
                for f in splits['folds'].keys():
                    t_s = time.time()
                    reviews_train = np.array(reviews)[splits['folds'][f]['train']].tolist()
                    reviews_train.extend([r_.augs[lang][1] for r_ in reviews_train for lang in params.settings['prep']['langaug'] if lang and r_.augs[lang][2] >= params.settings['train']['langaug_semsim']])
                    train(am, reviews_train, np.array(reviews)[splits['folds'][f]['valid']].tolist(), f, output, capability, cfg)
                    print(f'Trained time elapsed including language augs {cfg.prep.languag}: {time.time() - t_s}')

        # testing
        if 'test' in cfg.cmd:
            for capability in eval_for:
                for f in splits['folds'].keys():
                    test(am, np.array(reviews)[splits['test']].tolist(), f, output, capability)

        # evaluating
        if 'eval' in cfg.cmd:
            for capability in eval_for:
                print(f'Evaluating for {am.name} on {capability}')

                cp_name = get_capability_short_name(capability)

                df_f_means = pd.DataFrame()
                for f in splits['folds'].keys():
                    input = f'{output}f{f}.model.{cp_name}.pred.{params.settings["test"]["h_ratio"]}'
                    df_mean = evaluate(input, f'{input}.{cp_name}.eval.mean.csv', capability)
                    df_f_means = pd.concat([df_f_means, df_mean], axis=1)
                df_f_means.mean(axis=1).to_frame('mean').to_csv(f'{output}model.{cp_name}.pred.eval.mean.{params.settings["test"]["h_ratio"]}.csv')

# {CUDA_VISIBLE_DEVICES=0,1} won't work https://discuss.pytorch.org/t/using-torch-data-prallel-invalid-device-string/166233
# TOKENIZERS_PARALLELISM=true
# TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0 python -u main.py -am lda -naspect 5 -data ../data/raw/semeval/SemEval-14/Laptop_Train_v2.xml -output ../output/SemEval-14/Laptop 2>&1 | tee ../output/SemEval-14/Laptop/log.txt &
# TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0 python -u main.py -am lda -naspect 5 -data ../data/raw/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml -output ../output/toy.2016SB5 2>&1 | tee ../output/toy.2016SB5/log.txt &
# TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0 python -u main.py -am lda -naspect 5 -data ../data/raw/semeval/SemEval-14/Semeval-14-Restaurants_Train.xml -output ../output/SemEval-14/Restaurants 2>&1 | tee ../output/SemEval-14/Restaurants/log.txt &
# TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0 python -u main.py -am lda -naspect 5 -data ../data/raw/semeval/2015SB12/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml -output ../output/2015SB12 2>&1 | tee ../output/semeval+/2015SB12/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml/log.txt &
# TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0 python -u main.py -am lda -naspect 5 -data ../data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml -output ../output/2016SB5 2>&1 | tee ../output/2016SB5/log.txt &

if __name__ == '__main__':
    main()
    #if 'agg' in params.settings['cmd']: agg(cfg.args.output, cfg.args.output)