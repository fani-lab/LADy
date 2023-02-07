import argparse, os, pickle, multiprocessing, json
from time import time
import numpy as np
from json import JSONEncoder
import pandas as pd
import pytrec_eval

import params, visualization


from aml.mdl import AbstractAspectModel
from aml.lda import Lda
from aml.btm import Btm
from aml.rnd import Rnd


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
        # what is the type of input dataset?
        if input.endswith('.xml'):
            reviews = SemEvalReview.xmlloader(input)
        else:
            reviews = SemEvalReview.txtloader(input)
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


def evaluate(am, am_type, test):
    print(f'\n3. Evaluating on test set ...')
    print('#' * 50)
    pairs = []
    for r in test:
        r_aspects = [[w for a, o, s in sent for w in a] for sent in r.get_aos()]  # [['service', 'food'], ['service'], ...]
        r_ = r.hide_aspects()
        r_pred_aspects = am.infer(params.doctype, r_)
        if am_type == "btm":
            for i, subr_pred_aspects in enumerate(r_pred_aspects):
                subr_pred_aspects_words = [w_p for l in [[(w, a_p * w_p) for w, w_p in am.show_topic(a, 100)] for a, a_p in subr_pred_aspects] for w_p in l]
                subr_pred_aspects_words = sorted(subr_pred_aspects_words, reverse=True, key=lambda t: t[1])
                pairs.append((r_aspects[i], subr_pred_aspects_words))
            pass
        elif am_type == "rnd":
            for i in range(len(r_aspects)):
                pairs.append((r_aspects[i], r_pred_aspects))
        else:  # "lda" in am_type:
            for i, subr_pred_aspects in enumerate(r_pred_aspects):
                subr_pred_aspects_words = [w_p for l in [[(w, a_p * w_p) for w, w_p in am.mdl.show_topic(a, topn=100)] for a, a_p in subr_pred_aspects] for w_p in l]
                subr_pred_aspects_words = sorted(subr_pred_aspects_words, reverse=True, key=lambda t: t[1])
                # removing duplicate aspect words ==> handled in metrics()
                pairs.append((r_aspects[i], subr_pred_aspects_words))

    metrics_set = set()
    for m in params.metrics:
        metrics_set.add(f'{m}_{params.topkstr}')
    qrel = dict(); run = dict()
    print(f'3.1. Building pytrec_eval input for {len(pairs)} instances ...')
    for i, pair in enumerate(pairs):
        qrel['q' + str(i)] = {w: 1 for w in pair[0]}
        # the prediction list may have duplicates
        run['q' + str(i)] = {}
        if "btm" in am_type or "lda" in am_type:
            # print(pair[1])
            for j, (w, p) in enumerate(pair[1]):
                if w not in run['q' + str(i)].keys(): run['q' + str(i)][w] = len(pair[1]) - j
        else:  # rnd
            # print(pair[1])
            for j in range(len(pair[1])):
                # print(pair[1][j])
                run['q' + str(i)][pair[1][j]] = len(pair[1]) - j
    print(f'3.2. Calling pytrec_eval for {metrics_set} ...')
    df = pd.DataFrame.from_dict(pytrec_eval.RelevanceEvaluator(qrel, metrics_set).evaluate(run))
    print(f'3.3. Averaging ...')
    df_mean = df.mean(axis=1).to_frame('mean')
    return df_mean


def main(args):
    am_type = args.aml
    if not os.path.isdir(f'{args.output}/{args.naspects}'): os.makedirs(f'{args.output}/{args.naspects}')
    # output = f'{args.output}/{args.naspects}'
    reviews = load(args.data, args.output)
    splits = split(len(reviews), args.output)
    test = np.array(reviews)[splits['test']].tolist()
    for a in am_type:
        fold_mean = pd.DataFrame()
        output = f'{args.output}/{args.naspects}'
        print(a)
        if a == "btm":
            output = f'{output}/btm/'
            if not os.path.isdir(output): os.makedirs(output)
        elif a == "rnd":
            output = f'{output}/rnd/'
            if not os.path.isdir(output): os.makedirs(output)
        else:  # if am_type == "lda"
            output = f'{output}/lda/'
            if not os.path.isdir(output): os.makedirs(output)
        for f in splits['folds'].keys():
            output_ = f'{output}f{f}.'
            model_review = np.array(reviews)[splits['folds'][f]['train']].tolist()
            if a == "btm":
                am = Btm(model_review, args.naspects, params.no_extremes, output_)
            elif a == "rnd":
                am = Rnd(model_review, args.naspects, params.no_extremes, output_)
            else:  # if am_type == "lda"
                am = Lda(model_review, args.naspects, params.no_extremes, output_)

            # training
            print(f'\n2. Aspect modeling ...')
            print('#' * 50)
            t_s = time()
            try:
                print(f'2.1. Loading saved aspect model from {output} ...')
                am.load()
            except (FileNotFoundError, EOFError) as e:
                print(
                    f'2.1. Loading saved aspect model failed! Training a model for {args.naspects} of aspects. See {output_}model.train.log for training logs ...')
                am.train(params.doctype, multiprocessing.cpu_count() if params.cores <= 0 else params.cores, params.iter_c,
                         params.seed)
            print(f'2.2. Quality of aspects ...')
            for q in params.qualities:
                print(f'({q}: {AbstractAspectModel.quality(am, q)})')
            print(f'Time elapsed: {(time() - t_s)}')
            df_mean = evaluate(am, a, test)
            df_mean.to_csv(f'{output_}pred.eval.mean.csv')
            fold_mean = pd.concat([fold_mean, df_mean], axis=1)
        fold_mean.mean(axis=1).to_frame('mean').to_csv(f'{output}/pred.eval.mean.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latent Aspect Detection')
    parser.add_argument('--aml', '--aml-method-list', nargs='+', type=str.lower, required=True, help='a list of aspect modeling methods (eg. --aml lda rnd btm)')
    parser.add_argument('--data', dest='data', type=str, default='data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml', help='raw dataset file path, e.g., ..data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml')
    parser.add_argument('--output', dest='output', type=str, default='output/semeval/xml-2016', help='output path, e.g., ../output/semeval/xml-2016')
    parser.add_argument('--naspects', dest='naspects', type=int, default=25, help='user defined number of aspects.')
    args = parser.parse_args()

    main(args)

    # for p in ['../data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml', '../data/raw/semeval/2016.txt']:
    #     args.aml = ['btm', 'lda', 'rnd']
    #     args.data = p
    #     if str(p).endswith('.xml'):
    #         args.output = f'../output/semeval-2016-full/xml-version'
    #     else:
    #         args.output = f'../output/semeval-2016-full/txt-version'
    #     topic_range = range(1, 51, 1)
    #     for naspects in topic_range:
    #         args.naspects = naspects
    #         main(args)
    #     # visualization.plots_2d(args.output, 100, len(params.metrics), topic_range)
    #     visualization.plots_3d(args.output, topic_range)

