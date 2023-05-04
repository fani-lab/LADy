import argparse, os, pickle, multiprocessing, json, time
from tqdm import tqdm
import numpy as np, random, pandas as pd

import pytrec_eval
from nltk.corpus import wordnet as wn
# import nltk
# nltk.download('omw-1.4')

import params
# import visualization
from cmn.review import Review

def load(input, output):
    print('\n1. Loading reviews and preprocessing ...')
    print('#' * 50)
    try:
        print(f'1.1. Loading existing processed reviews file {output}...')
        with open(f'{output}', 'rb') as f: reviews = pickle.load(f)
    except (FileNotFoundError, EOFError) as e:
        print('1.1. Loading existing processed pickle file failed! Loading raw reviews ...')
        from cmn.semeval import SemEvalReview
        if str(input).endswith('.xml'): reviews = SemEvalReview.xmlloader(input)
        else: reviews = SemEvalReview.txtloader(input)
        print(f'(#reviews: {len(reviews)})')
        print(f'\n1.2. Augmentation via backtranslation by {params.settings["prep"]["langaug"]} {"in batches" if params.settings["prep"] else ""}...')
        for lang in params.settings['prep']['langaug']:
            print(f'\n{lang} ...')
            if params.settings["prep"]['batch']:
                start = time.time()
                Review.translate_batch(reviews, lang, params.settings['prep']) #all at once, esp., when using gpu
                end = time.time()
                print(f'{lang} done all at once (batch). Time: {end - start}')
            else:
                for r in tqdm(reviews): r.translate(lang, params.settings['prep'])
        print(f'\n1.3. Saving processed pickle file {output}...')
        with open(output, 'wb') as f: pickle.dump(reviews, f, protocol=pickle.HIGHEST_PROTOCOL)
    return reviews

def split(nsample, output):
    # We split originals into train, valid, test. So each have its own augmented versions.
    # During test (or even train), we can decide to consider augmented version or not.

    from sklearn.model_selection import KFold, train_test_split
    from json import JSONEncoder

    train, test = train_test_split(np.arange(nsample), train_size=params.settings['train']['train_ratio'], random_state=params.seed, shuffle=True)

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

def train(args, am, train, valid, f, output):
    try:
        print(f'2.1. Loading saved aspect model from {output} ...')
        am.load(f'{output}/f{f}.')
    except (FileNotFoundError, EOFError) as e:
        print(f'2.1. Loading saved aspect model failed! Training {am.__class__.__name__.lower()} for {args.naspects} of aspects. See {output}/f{f}.model.train.log for training logs ...')
        if not os.path.isdir(output): os.makedirs(output)
        am.train(train, valid, params.settings['train'][args.am], params.settings['prep']['doctype'], params.settings['prep']['langaug'], f'{output}/f{f}.')

        from aml.mdl import AbstractAspectModel
        print(f'2.2. Quality of aspects ...')
        for q in params.settings['train'][args.am]['qualities']: print(f'({q}: {AbstractAspectModel.quality(am, q)})')

def test(am, test, f, output):
    try:
        print(f'\n3. Loading saved predictions on test set from {output}f{f}.model.pred.{params.settings["test"]["h_ratio"]} ...')
        with open(f'{output}f{f}.model.pred.{params.settings["test"]["h_ratio"]}', 'rb') as f: return pickle.load(f)
    except (FileNotFoundError, EOFError) as e:
        print(f'\n3. Loading saved predictions on test set failed! Predicting on the test set with {params.settings["test"]["h_ratio"] * 100}% latent aspect ...')
        print('#' * 50)

        print(f'3.1. Loading saved aspect model from {output}f{f}.model ...')
        am.load(f'{output}/f{f}.')

        # # ???
        # if am.__class__.__name__.lower() == 'nrl': #isinstance(am, Nrl):
        #     with open(f'{output}ridx.pkl', 'rb') as f: ridx = pickle.load(f)
        # ######

        pairs = []
        r_list = []
        r_aspect_list = []

        # Since predicted aspects are distributions over words, we need to flatten them into list of words.
        # Given a and b are two aspects, we do prob(a) * prob(a_w) for all w \in a and prob(b) * prob(b_w) for all w \in b
        # Then sort.
        def rank_pairs(r_aspects, r_pred_aspects):
            nwords = params.settings['train'][am.__class__.__name__.lower()]['nwords']
            for i, subr_pred_aspects in enumerate(r_pred_aspects):
                subr_pred_aspects_words = [w_p for l in [[(w, a_p * w_p) for w, w_p in am.get_aspect(a, nwords)] for a, a_p in subr_pred_aspects] for w_p in l]
                subr_pred_aspects_words = sorted(subr_pred_aspects_words, reverse=True, key=lambda t: t[1])
                # removing duplicate aspect words ==> handled in metrics()
                pairs.append((r_aspects[i], subr_pred_aspects_words))

        idx = 0
        for r_idx, r in enumerate(test):
            r_aspects = [[w for a, o, s in sent for w in a] for sent in r.get_aos()]  # [['service', 'food'], ['service'], ...]
            if len(r_aspects[0]) == 0: continue #??
            if random.random() < params.settings['test']['h_ratio']: r_ = r.hide_aspects()
            else: r_ = r
            r_list.append(r_)
            r_aspect_list.append(r_aspects)

            # # ???
            # if am.__class__.__name__.lower() == 'ctm': continue #see below
            # elif am.__class__.__name__.lower() == 'nrl':
            #     if r_idx in ridx: continue
            #     r_pred_aspects = am.infer(params.settings['prep']['doctype'], r_, idx)
            # ######
            # else:
            r_pred_aspects = am.infer(params.settings['prep']['doctype'], r_)
            rank_pairs(r_aspects, r_pred_aspects)
            idx += 1

        # # ???
        # if am.__class__.__name__.lower() == 'ctm':
        #     r_pred_aspects = am.infer(params.settings['prep']['doctype'], r_list)
        #     r_aspect_list_extended = []
        #     for r in r_aspect_list: r_aspect_list_extended.extend(r)
        #     rank_pairs(r_aspect_list_extended, r_pred_aspects)
        # ######

        with open(f'{output}f{f}.model.pred.{params.settings["test"]["h_ratio"]}', 'wb') as f: pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
        return pairs

def evaluate(input, output):
    with open(input, 'rb') as f: pairs = pickle.load(f)
    metrics_set = set(f'{m}_{",".join([str(i) for i in params.settings["eval"]["topkstr"]])}' for m in params.settings['eval']['metrics'])
    qrel = dict(); run = dict()
    print(f'3.1. Building pytrec_eval input for {len(pairs)} instances ...')
    for i, pair in enumerate(pairs):
        if params.settings['eval']['syn']:
            syn_list = set()
            for p_instance in pair[0]:
                syn_list.add(p_instance)
                syn_list.update(set([lemma.name() for syn in wn.synsets(p_instance) for lemma in syn.lemmas()]))
            qrel['q' + str(i)] = {w: 1 for w in syn_list}
        else: qrel['q' + str(i)] = {w: 1 for w in pair[0]}

        # the prediction list may have duplicates
        run['q' + str(i)] = {}
        for j, (w, p) in enumerate(pair[1]):
            if w not in run['q' + str(i)].keys(): run['q' + str(i)][w] = len(pair[1]) - j

    print(f'3.2. Calling pytrec_eval for {metrics_set} ...')
    df = pd.DataFrame.from_dict(pytrec_eval.RelevanceEvaluator(qrel, metrics_set).evaluate(run))  # qrel should not have empty entry otherwise get exception
    print(f'3.3. Averaging ...')
    df_mean = df.mean(axis=1).to_frame('mean')
    df_mean.to_csv(output)
    return df_mean

def agg(path, output):
    files = list()
    for dirpath, dirnames, filenames in os.walk(path): files += [os.path.join(os.path.normpath(dirpath), file).split(os.sep) for file in filenames if file.startswith("model.pred.eval.mean")]

    column_names = []
    for f in files:
        p = ".".join(f[-3:]).replace('.csv', '').replace('model.pred.eval.mean.', '')
        column_names.append(p)
    column_names.insert(0, 'metric')

    all_results = pd.DataFrame()
    for i, f in enumerate(files):
        df = pd.read_csv(os.sep.join(f))
        if i == 0: all_results = df
        else: all_results = pd.concat([all_results, df['mean']], axis=1)

    all_results.columns = column_names
    all_results.to_csv(f'{output}/agg.pred.eval.mean.csv', index=False)
    return all_results

def main(args):
    if not os.path.isdir(args.output): os.makedirs(args.output)
    reviews = load(args.data, f'{args.output}/reviews.{".".join(params.settings["prep"]["langaug"])}.pkl'.replace('..pkl', '.pkl'))

    print(f'\n2. Aspect modeling for {args.am} ...')
    if not os.path.isdir(f'{args.output}/{args.naspects}'): os.makedirs(f'{args.output}/{args.naspects}')
    if "rnd" == args.am: from aml.rnd import Rnd; am = Rnd(args.naspects)
    if "lda" == args.am: from aml.lda import Lda; am = Lda(args.naspects)
    if "btm" == args.am: from aml.btm import Btm; am = Btm(args.naspects)
    if "ctm" == args.am: from aml.ctm import Ctm; am = Ctm(args.naspects)
    if "nrl" == args.am: from aml.nrl import Nrl; am = Nrl(args.naspects)

    output = f'{args.output}/{args.naspects}/{am.__class__.__name__.lower()}/'
    splits = split(len(reviews), args.output)
    if 'train' in params.settings['cmd']:
        for f in splits['folds'].keys():
            t_s = time.time()
            train(args, am, np.array(reviews)[splits['folds'][f]['train']].tolist(), np.array(reviews)[splits['folds'][f]['valid']].tolist(), f, output)
            print(f'Time elapsed: {time.time() - t_s}')

    # testing
    if 'test' in params.settings['cmd']:
        for f in splits['folds'].keys(): pairs = test(am, np.array(reviews)[splits['test']].tolist(), f, output)

    # evaluating
    if 'eval' in params.settings['cmd']:
        df_f_means = pd.DataFrame()
        for f in splits['folds'].keys():
            input = f'{output}f{f}.model.pred.{params.settings["test"]["h_ratio"]}'
            df_mean = evaluate(input, f'{input}.eval.mean.csv')
            df_f_means = pd.concat([df_f_means, df_mean], axis=1)
        df_f_means.mean(axis=1).to_frame('mean').to_csv(f'{output}model.pred.eval.mean.{params.settings["test"]["h_ratio"]}.csv')

# python -u main.py -am lda -data ../data/raw/semeval/toy.2016.txt -output ../output/semeval/toy.2016 -naspect 25
# python -u main.py -am lda -data ../data/raw/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml -output ../output/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2 -naspect 25

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latent Aspect Detection')
    parser.add_argument('-am', type=str.lower, required=True, help='aspect modeling method (eg. --am lda)')
    parser.add_argument('-data', dest='data', type=str, help='raw dataset file path, e.g., -data ..data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml')
    parser.add_argument('-output', dest='output', type=str, default='../output/', help='output path, e.g., -output ../output/semeval/2016.xml')
    parser.add_argument('-naspects', dest='naspects', type=int, default=25, help='user-defined number of aspects, e.g., -naspect 25')
    args = parser.parse_args()

    # main(args)

    # to run pipeline for all available aspect modeling methods
    datasets = [('../data/raw/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml', '../output/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml'),
                ('../data/raw/semeval/SemEval-14/Semeval-14-Restaurants_Train.xml', '../output/semeval/SemEval-14/Semeval-14-Restaurants_Train.xml'),
                ('../data/raw/semeval/2015SB12/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml', '../output/semeval/2015SB12/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml'),
                ('../data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml', '../output/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml')]

    for (data, output) in datasets:
        for am in ['rnd', 'lda', 'btm']:#, 'rnd', 'lda', 'btm', 'ctm', 'nrl']:
            for naspects in range(5, 30, 5):
                for hide in range(0, 110, 10):
                    args.am = am
                    args.data = data
                    args.output = output
                    args.naspects = naspects
                    params.settings['test']['h_ratio'] = round(hide * 0.01, 1)
                    main(args)

    if 'agg' in params.settings['cmd']: agg(args.output, args.output)



