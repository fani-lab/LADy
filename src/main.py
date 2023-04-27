import argparse, os, pickle, multiprocessing, json, time
from tqdm import tqdm
# import numpy as np
# from json import JSONEncoder
# import pandas as pd
# import pytrec_eval
# from nltk.corpus import wordnet as wn
# import random

import params
# import visualization
from cmn.review import Review
# from aml.mdl import AbstractAspectModel
# from aml.lda import Lda
# from aml.btm import Btm
# from aml.rnd import Rnd
# from aml.neural import Neural
# from aml.ctm import CTM

def load(input, output):
    print('\n1. Loading reviews and preprocessing ...')
    print('#' * 50)
    try:
        print(f'\n1.1. Loading existing processed reviews file {output}...')
        with open(f'{output}', 'rb') as f: reviews = pickle.load(f)
    except (FileNotFoundError, EOFError) as e:
        print('\n1.1. Loading existing processed pickle file failed! Loading raw reviews ...')
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

# def split(nsample, output):
#     from sklearn.model_selection import KFold, train_test_split
#     train, test = train_test_split(np.arange(nsample), train_size=params.train_ratio, random_state=params.seed,
#                                    shuffle=True)
#
#     splits = dict()
#     splits['test'] = test
#     splits['folds'] = dict()
#     skf = KFold(n_splits=params.nfolds, random_state=params.seed, shuffle=True)
#     for k, (trainIdx, validIdx) in enumerate(skf.split(train)):
#         splits['folds'][k] = dict()
#         splits['folds'][k]['train'] = train[trainIdx]
#         splits['folds'][k]['valid'] = train[validIdx]
#
#     class NumpyArrayEncoder(JSONEncoder):
#         def default(self, obj):
#             if isinstance(obj, np.ndarray): return obj.tolist()
#             return JSONEncoder.default(self, obj)
#
#     with open(f'{output}/splits.json', 'w') as f:
#         json.dump(splits, f, cls=NumpyArrayEncoder, indent=1)
#     return splits


# def inference(am, am_type, test, hide_percentage, output_):
#     if am_type == 'neural':
#         with open(f'{output_}ridx.pkl', 'rb') as f:
#             ridx = pickle.load(f)
#     print(f'\n3. Evaluating on test set ...')
#     print('#' * 50)
#     pairs = []
#     r_list = []
#     r_aspect_list = []
#     idx = 0
#     for r_idx, r in enumerate(test):
#         r_aspects = [[w for a, o, s in sent for w in a] for sent in
#                      r.get_aos()]  # [['service', 'food'], ['service'], ...]
#         if len(r_aspects[0]) == 0:
#             continue
#         if random.random() < hide_percentage:
#             r_ = r.hide_aspects()
#         else:
#             r_ = r
#         r_list.append(r_)
#         r_aspect_list.append(r_aspects)
#         if am_type == "ctm":
#             continue
#         elif am_type == 'neural':
#             if r_idx in ridx:
#                 continue
#             r_pred_aspects = am.infer(params.doctype, r_, idx)
#         else:
#             r_pred_aspects = am.infer(params.doctype, r_)
#
#         if am_type == "btm" or am_type == "neural":
#             for i, subr_pred_aspects in enumerate(r_pred_aspects):
#                 subr_pred_aspects_words = [w_p for l in
#                                            [[(w, a_p * w_p) for w, w_p in am.show_topic(a, params.nwords)] for a, a_p in
#                                             subr_pred_aspects] for w_p in l]
#                 subr_pred_aspects_words = sorted(subr_pred_aspects_words, reverse=True, key=lambda t: t[1])
#                 pairs.append((r_aspects[i], subr_pred_aspects_words))
#             pass
#         elif am_type == "rnd":
#             for i in range(len(r_aspects)):
#                 pairs.append((r_aspects[i], r_pred_aspects))
#         else:  # "lda" in am_type:
#             for i, subr_pred_aspects in enumerate(r_pred_aspects):
#                 subr_pred_aspects_words = [w_p for l in
#                                            [[(w, a_p * w_p) for w, w_p in am.mdl.show_topic(a, topn=params.nwords)] for
#                                             a, a_p in subr_pred_aspects] for w_p in l]
#                 subr_pred_aspects_words = sorted(subr_pred_aspects_words, reverse=True, key=lambda t: t[1])
#                 # removing duplicate aspect words ==> handled in metrics()
#                 pairs.append((r_aspects[i], subr_pred_aspects_words))
#         idx += 1
#
#     if am_type == "ctm":
#         r_pred_aspects = am.infer(params.doctype, r_list)
#         r_aspect_list_extended = []
#         for r in r_aspect_list:
#             r_aspect_list_extended.extend(r)
#         for i, subr_pred_aspects in enumerate(r_pred_aspects):
#             subr_pred_aspects_words = [w_p for l in
#                                        [[(w, a_p * w_p) for w, w_p in am.show_topic(a, params.nwords)] for a, a_p in
#                                         subr_pred_aspects] for w_p in l]
#             subr_pred_aspects_words = sorted(subr_pred_aspects_words, reverse=True, key=lambda t: t[1])
#             pairs.append((r_aspect_list_extended[i], subr_pred_aspects_words))
#     return pairs


# def evaluate(am_type, syn_status, pairs):
#     metrics_set = set()
#     for m in params.metrics:
#         metrics_set.add(f'{m}_{params.topkstr}')
#     qrel = dict();
#     run = dict()
#     print(f'3.1. Building pytrec_eval input for {len(pairs)} instances ...')
#     for i, pair in enumerate(pairs):
#         if syn_status == 'yes':
#             syn_list = set()
#             for p_instance in pair[0]:
#                 syn_list.add(p_instance)
#                 for syn in wn.synsets(p_instance):
#                     for lemma in syn.lemmas():
#                         syn_list.add(lemma.name())
#             qrel['q' + str(i)] = {w: 1 for w in syn_list}
#         else:
#             qrel['q' + str(i)] = {w: 1 for w in pair[0]}
#         # the prediction list may have duplicates
#         run['q' + str(i)] = {}
#         if "btm" in am_type or "lda" in am_type or "neural" in am_type or "ctm" in am_type:
#             # print(pair[1])
#             for j, (w, p) in enumerate(pair[1]):
#                 if w not in run['q' + str(i)].keys(): run['q' + str(i)][w] = len(pair[1]) - j
#         else:  # rnd
#             # print(pair[1])
#             for j in range(len(pair[1])):
#                 # print(pair[1][j])
#                 run['q' + str(i)][pair[1][j]] = len(pair[1]) - j
#     print(f'3.2. Calling pytrec_eval for {metrics_set} ...')
#     df = pd.DataFrame.from_dict(pytrec_eval.RelevanceEvaluator(qrel, metrics_set).evaluate(
#         run))  # qrel should not have empty entry otherwise get exception
#     print(f'3.3. Averaging ...')
#     df_mean = df.mean(axis=1).to_frame('mean')
#     return df_mean
#
#
# def load_bt(dataset_path):
#     # df = Review.save_sentences(reviews, path)
#     dataset_df = pd.read_csv(f'{dataset_path}')
#     bt_reviews = dataset_df['sentences'].tolist()
#     bt_labels = dataset_df['labels'].tolist()
#     reviews_list = []
#     for i in range(len(bt_reviews)):
#         reviews_list.append(
#             Review(id=i, sentences=[[str(t).lower() for t in bt_reviews[i].split()]], time=None,
#                    author=None, aos=[[(label, [], 0) for label in eval(bt_labels[i])]], lempos=""))
#
#     return reviews_list
#
#
# def aggregate(path, save_path, naspects):
#     if os.path.isdir(path):
#         # given a root folder, we can crawl the folder to find pred files
#         files = list()
#         for dirpath, dirnames, filenames in os.walk(path): files += [
#             os.path.join(os.path.normpath(dirpath), file).split(os.sep) for file in filenames if
#             file.startswith("pred")]
#
#     column_names = []
#     for i in range(len(files)):
#         p = ".".join(files[i]).replace('.csv', '').replace('...output.', '')
#         column_names.append(p)
#     column_names.insert(0, 'metric')
#
#     f_path = []
#     all_results = pd.DataFrame()
#     for i in range(len(files)):
#         p = "\\".join(files[i])
#         f_path.append(p)
#         df = pd.read_csv(p)
#         if i == 0:
#             all_results = df
#         else:
#             all_results = pd.concat([all_results, df['mean']], axis=1)
#
#     all_results.columns = column_names
#     all_results.to_csv(f'{save_path}/{naspects}aspects.agg.pred.eval.mean.csv')
#     return all_results


def main(args):
    if not os.path.isdir(args.output): os.makedirs(args.output)
    reviews = load(args.data, f'{args.output}/reviews.{".".join(params.settings["prep"]["langaug"])}.pkl'.replace('..pkl', '.pkl'))
    return
    #
    # bt_reviews = load_bt(args.btdata)
    # reviews = org_reviews + bt_reviews
    #
    # am_type = args.aml
    # if not os.path.isdir(f'{args.output}/{args.naspects}'): os.makedirs(f'{args.output}/{args.naspects}')
    #
    # # reviews = load(args.data, args.output)
    # splits = split(len(reviews), args.output)
    # test = np.array(reviews)[splits['test']].tolist()
    # for a in am_type:
    #     fold_mean_list = [pd.DataFrame() for i in range(0, 11)]
    #     output = f'{args.output}/{args.naspects}'
    #     print(a)
    #     if a == "btm":
    #         output = f'{output}/btm/'
    #         if not os.path.isdir(output): os.makedirs(output)
    #     elif a == "rnd":
    #         output = f'{output}/rnd/'
    #         if not os.path.isdir(output): os.makedirs(output)
    #     elif a == "lda":
    #         output = f'{output}/lda/'
    #         if not os.path.isdir(output): os.makedirs(output)
    #     elif a == "ctm":
    #         output = f'{output}/ctm/'
    #         if not os.path.isdir(output): os.makedirs(output)
    #     else:  # if am_type == "neural"
    #         output = f'{output}/neural/'
    #         if not os.path.isdir(output): os.makedirs(output)
    #     for f in splits['folds'].keys():
    #         output_ = f'{output}f{f}.'
    #         model_review = np.array(reviews)[splits['folds'][f]['train']].tolist()
    #         if a == "btm":
    #             am = Btm(model_review, args.naspects, params.no_extremes, output_)
    #         elif a == "rnd":
    #             am = Rnd(model_review, args.naspects, params.no_extremes, output_)
    #         elif a == "lda":
    #             am = Lda(model_review, args.naspects, params.no_extremes, output_)
    #         elif a == "ctm":
    #             am = CTM(model_review, args.naspects, params.no_extremes, output_)
    #         else:  # if am_type == "neural"
    #             am = Neural(model_review, args.naspects, params.no_extremes, output_)
    #         # training
    #         print(f'\n2. Aspect modeling ...')
    #         print('#' * 50)
    #         t_s = time()
    #         try:
    #             print(f'2.1. Loading saved aspect model from {output} ...')
    #             am.load()
    #         except (FileNotFoundError, EOFError) as e:
    #             print(
    #                 f'2.1. Loading saved aspect model failed! Training a model for {args.naspects} of aspects. See {output_}model.train.log for training logs ...')
    #             if a == 'neural':
    #                 am.train(params.doctype, multiprocessing.cpu_count() if params.cores <= 0 else params.cores,
    #                          params.iter_c, params.seed, test)
    #             else:
    #                 am.train(params.doctype, multiprocessing.cpu_count() if params.cores <= 0 else params.cores,
    #                          params.iter_c, params.seed)
    #         print(f'2.2. Quality of aspects ...')
    #         for q in params.qualities:
    #             print(f'({q}: {AbstractAspectModel.quality(am, q)})')
    #         print(f'Time elapsed: {(time() - t_s)}')
    #
    #         for i in range(0, 11):
    #             hide_percentage = i * 10
    #             hp = i / 10
    #             pairs = inference(am, a, test, hp, output_)
    #             df_mean = evaluate(a, args.syn, pairs)
    #             df_mean.to_csv(f'{output_}pred.eval.mean.{hide_percentage}.csv')
    #             fold_mean_list[i] = pd.concat([fold_mean_list[i], df_mean], axis=1)
    #     for i in range(0, 11):
    #         hide_percentage = i * 10
    #         fold_mean_list[i].mean(axis=1).to_frame('mean').to_csv(f'{output}/pred.eval.mean.{hide_percentage}.csv')

# python -u main.py -aml lda -data ../data/raw/semeval/toy.2016.txt -output ../output/semeval/toy.2016 -naspect 25
# python -u main.py -aml lda -data ../data/raw/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml -output ../output/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2 -naspect 25

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latent Aspect Detection')
    parser.add_argument('-aml', nargs='+', type=str.lower, required=True, help='a list of aspect modeling methods (eg. --aml lda rnd btm neural ctm)')
    parser.add_argument('-data', dest='data', type=str, help='raw dataset file path, e.g., -data ..data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml')
    parser.add_argument('-output', dest='output', type=str, default='../output/', help='output path, e.g., -output ../output/semeval/2016.xml')
    parser.add_argument('-naspects', dest='naspects', type=int, default=25, help='user-defined number of aspects, e.g., -naspect 25')

    args = parser.parse_args()

    main(args)

    #generate seperate review pickle file for each lang including empty (no lang)
    import copy
    langs = copy.deepcopy(params.settings['prep']['langaug'])
    for l in langs:
        params.settings['prep']['langaug'] = [l]
        main(args)
    params.settings['prep']['langaug'] = []
    main(args)

    # aggregate(path=args.output, save_path=f'{args.output}/{args.naspects}', naspects=args.naspects)
    #
    # for lang in ['Chinese', 'German', 'Arabic', 'French', 'Spanish', 'All']:
    #     if lang == 'Chinese':
    #         args.btdata = '../output/augmentation-R-16/augmented-with-labels/Chinese.back-translated.with-labels.csv'
    #     elif lang == 'German':
    #         args.btdata = '../output/augmentation-R-16/augmented-with-labels/German.back-translated.with-labels.csv'
    #     elif lang == 'Arabic':
    #         args.btdata = '../output/augmentation-R-16/augmented-with-labels/Arabic.back-translated.with-labels.csv'
    #     elif lang == 'French':
    #         args.btdata = '../output/augmentation-R-16/augmented-with-labels/French.back-translated.with-labels.csv'
    #     elif lang == 'Spanish':
    #         args.btdata = '../output/augmentation-R-16/augmented-with-labels/Spanish.back-translated.with-labels.csv'
    #     else:  # 'All'
    #         args.btdata = '../output/augmentation-R-16/augmented-with-labels/All.back-translated.with-labels.csv'
    #     args.aml = ['rnd', 'btm', 'lda', 'neural']
    #     args.data = '../data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml'
    #     args.output = f'../output/{lang}/Semeval-2016+'
    #     args.syn = 'yes'
    #     topic_range = range(0, 51, 5)
    #     for naspects in topic_range:
    #         if naspects == 0:
    #             naspects = 1
    #         args.naspects = naspects
    #         main(args)
    #         aggregate(path=args.output, save_path=f'{args.output}/{args.naspects}', naspects=args.naspects)

    # for p in ['../data/raw/semeval/2016SB5/`ABSA16_Restaurants_Train_SB1_v2.xml', '../data/raw/semeval/2016.txt']:
    #     args.aml = ['btm', 'lda', 'rnd', 'ctm']
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



