import argparse, os, pickle, multiprocessing, json
from time import time
import numpy as np
from json import JSONEncoder
import pandas as pd
import pytrec_eval


import params

def load(input, output):
    print('\n1. Loading reviews and preprocessing ...')
    print('#' * 50)
    t_s = time()
    try:
        print('\n1.1. Loading existing processed reviews file ...')
        with open(f'{output}/reviews.pkl', 'rb') as f: reviews = pickle.load(f)
    except (FileNotFoundError, EOFError) as e:
        print('\n1.1. Loading existing processed pickle file failed! Loading raw reviews ...')
        from cmn.semeval import SemEvalReview
        reviews = SemEvalReview.load(input)
        print('\n1.2. Saving processed pickle file ...')
        with open(f'{output}/reviews.pkl', 'wb') as f: pickle.dump(reviews, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'(#reviews: {len(reviews)})')
    print(f'Time elapsed: {(time() - t_s)}')
    return reviews


def split(nsample, output):
    from sklearn.model_selection import KFold, train_test_split
    train, test = train_test_split(np.arange(nsample), train_size=params.train_ratio, random_state=params.seed, shuffle=True)

    splits = dict()
    splits['test'] = test
    splits['folds'] = dict()
    skf = KFold(n_splits=params.nfolds, random_state=params.seed, shuffle=True)
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

def train(reviews, naspects, output):
    print(f'\n2. Aspect modeling ...')
    print('#' * 50)
    t_s = time()
    from aml.lda import AspectModel
    try:
        print(f'2.1. Loading saved aspect model from {output} ...')
        am = AspectModel(reviews, naspects, params.no_extremes, output)
        am.load()
    except (FileNotFoundError, EOFError) as e:
        print(f'2.1. Loading saved aspect model failed! Training a model for {naspects} of aspects. See {output}model.train.log for training logs ...')
        am.train(params.doctype, multiprocessing.cpu_count() if params.cores <= 0 else params.cores, params.iter, params.seed)
    print(f'2.2. Quality of aspects ...')
    print(f'(Coherence: {np.mean(am.cas)}\u00B1{np.std(am.cas)})')
    print(f'Time elapsed: {(time() - t_s)}')
    return am

def metrics(pairs, per_instance=False, metrics={'success_1,2,5,10,100', 'P_1,2,5,10,100', 'recall_1,2,5,10,100', 'ndcg_cut_1,2,5,10,100', 'map_cut_1,2,5,10,100'}):
    qrel = dict(); run = dict()
    print(f'3.1. Building pytrec_eval input for {len(pairs)} instances ...')
    for i, pair in enumerate(pairs):
        qrel['q' + str(i)] = {w: 1 for w in pair[0]}
        # the prediction list may have duplicates
        run['q' + str(i)] = {}
        for j, (w, p) in enumerate(pair[1]):
            if w not in run['q' + str(i)].keys(): run['q' + str(i)][w] = len(pair[1]) - j
        # run['q' + str(i)] = {w: len(pair[1]) - i for i, (w, p) in enumerate(pair[1])}

    print(f'3.2. Calling pytrec_eval for {metrics} ...')
    df = pd.DataFrame.from_dict(pytrec_eval.RelevanceEvaluator(qrel, metrics).evaluate(run))
    print(f'3.3. Averaging ...')
    df_mean = df.mean(axis=1).to_frame('mean')
    return df if per_instance else None, df_mean

def evaluate(am, test):
    print(f'\n3. Evaluating on test set ...')
    print('#' * 50)
    pairs = []
    for r in test:
        r_aspects = [[w for a, o, s in sent for w in a] for sent in r.get_aos()]  # [['service', 'food'], ['service'], ...]
        r_ = r.hide_aspects()
        r_pred_aspects = am.infer(params.doctype, r_)
        for i, subr_pred_aspects in enumerate(r_pred_aspects):
            subr_pred_aspects_words = [w_p for l in [[(w, a_p * w_p) for w, w_p in am.mdl.show_topic(a, topn=100)] for a, a_p in subr_pred_aspects] for w_p in l]
            subr_pred_aspects_words = sorted(subr_pred_aspects_words, reverse=True, key=lambda t: t[1])
            # removing duplicate aspect words ==> handled in metrics()
            pairs.append((r_aspects[i], subr_pred_aspects_words))
    return metrics(pairs, per_instance=True)

def main(args):
    if not os.path.isdir(f'{args.output}/{args.naspects}'): os.makedirs(f'{args.output}/{args.naspects}')

    reviews = load(args.data, args.output)
    splits = split(len(reviews), args.output)
    test = np.array(reviews)[splits['test']].tolist()
    fold_mean = pd.DataFrame()
    output = f'{args.output}/{args.naspects}'
    for f in splits['folds'].keys():
        output_ = f'{output}/f{f}.'
        am = train(np.array(reviews)[splits['folds'][f]['train']].tolist(), args.naspects, output_)
        _, df_mean = evaluate(am, test)
        df_mean.to_csv(f'{output_}pred.eval.mean.csv')
        fold_mean = pd.concat([fold_mean, df_mean], axis=1)
    fold_mean.mean(axis=1).to_frame('mean').to_csv(f'{output}/pred.eval.mean.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PXP Topic Modeling.')
    parser.add_argument('--data', dest='data', type=str, default='../data/raw/semeval/2016.txt', help='raw dataset file path, e.g., ../data/raw/semeval-umass/2016.txt')
    parser.add_argument('--output', dest='output', type=str, default='../output/semeval/2016', help='output path, e.g., ../output/semeval2016')
    parser.add_argument('--naspects', dest='naspects', type=int, default=25, help='user defined number of aspects.')
    args = parser.parse_args()

    # main(args)

    for year in [2016, 2015, 2014]:
        for naspects in range(5, 55, 5):
            args.naspects = naspects
            args.data = f'../data/raw/semeval/{year}.txt'
            args.output = f'../output/semeval-cv-20221004/{year}'
            main(args)

