import argparse
import os
import pickle
import json
import numpy as np
from random import random
import re
import pandas as pd

from cmn.review import Review


def load(reviews, splits):
    print('\n Loading reviews and preprocessing ...')
    print('#' * 50)
    try:
        print('\nLoading reviews file ...')
        with open(f'{reviews}', 'rb') as f:
            reviews = pickle.load(f)
        with open(f'{splits}', 'r') as f:
            splits = json.load(f)
    except (FileNotFoundError, EOFError) as e:
        print(e)
        print('\nLoading existing file failed!')
    print(f'(#reviews: {len(reviews)})')
    return reviews, splits


def get_aos_augmented(review):
    r = []
    if not review.aos: return r
    for i, aos in enumerate([review.aos]): r.append(
        [([review.sentences[i][j] for j in a], [review.sentences[i][j] for j in o], s) for (a, o, s) in aos])
    return r


def preprocess(org_reviews, is_test, lang):
    reviews_list = []
    label_list = []
    for r in org_reviews:
        if not len(r.aos[0]):
            continue
        else:
            aos_list = []
            if r.augs and not is_test:
                if lang == 'pes_Arab.zho_Hans.deu_Latn.arb_Arab.fra_Latn.spa_Latn':
                    for key, value in r.augs.items():
                        aos_list = []
                        text = ' '.join(r.augs[key][1].sentences[0]).strip() + '####'
                        for aos_instance in r.augs[key][1].aos:
                            aos_list.extend(aos_instance[0])
                        for idx, word in enumerate(r.augs[key][1].sentences[0]):
                            if idx in aos_list:
                                tag = word + '=T-POS' + ' '
                                text += tag
                            else:
                                tag = word + '=O' + ' '
                                text += tag
                        if len(text.rstrip()) > 511:
                            continue
                        reviews_list.append(text.rstrip())
                else:
                    for aos_instance in r.augs[lang][1].aos:
                        aos_list.extend(aos_instance[0])
                    text = ' '.join(r.augs[lang][1].sentences[0]).strip() + '####'
                    for idx, word in enumerate(r.augs[lang][1].sentences[0]):
                        if idx in aos_list:
                            tag = word + '=T-POS' + ' '
                            text += tag
                        else:
                            tag = word + '=O' + ' '
                            text += tag
                    if len(text.rstrip()) > 511:
                        continue
                    reviews_list.append(text.rstrip())

            aos_list = []
            for aos_instance in r.aos[0]:
                aos_list.extend(aos_instance[0])
            # text = ' '.join(r.sentences[0]).replace('*****', '').replace('   ', '  ').replace('  ', ' ').replace('  ', ' ') + '####'
            # text = re.sub(r'\s{2,}', ' ', ' '.join(r.sentences[0]).replace('#####', '').strip()) + '####'
            text = re.sub(r'\s{2,}', ' ', ' '.join(r.sentences[0]).strip()) + '####'
            for idx, word in enumerate(r.sentences[0]):
                # if is_test and word == "#####":
                #     continue
                if idx in aos_list:
                    tag = word + '=T-POS' + ' '
                    text += tag
                else:
                    tag = word + '=O' + ' '
                    text += tag
            if len(text.rstrip()) > 511:
                continue
            reviews_list.append(text.rstrip())
            aos_list_per_review = []
            for idx, word in enumerate(r.sentences[0]):
                if idx in aos_list:
                    aos_list_per_review.append(word)
            label_list.append(aos_list_per_review)
    return reviews_list, label_list


# python main.py -ds_name [YOUR_DATASET_NAME] -sgd_lr [YOUR_LEARNING_RATE_FOR_SGD] -win [YOUR_WINDOW_SIZE] -optimizer [YOUR_OPTIMIZER] -rnn_type [LSTM|GRU] -attention_type [bilinear|concat]
def main(args):
    output_path = f'{args.output}/{args.dname}'
    print(output_path)
    # if not os.path.isdir(output_path): os.makedirs(output_path)
    org_reviews, splits = load(args.reviews, args.splits)

    test = np.array(org_reviews)[splits['test']].tolist()
    _, labels = preprocess(test, True, args.lang)
    for h in range(0, 101, 10):
        hp = h / 100

        for f in range(5):
            train, _ = preprocess(np.array(org_reviews)[splits['folds'][str(f)]['train']].tolist(), False, args.lang)
            dev, _ = preprocess(np.array(org_reviews)[splits['folds'][str(f)]['valid']].tolist(), False, args.lang)
            path = f'{output_path}-fold-{f}-latency-{h}'
            if not os.path.isdir(path): os.makedirs(path)

            with open(f'{path}/dev.txt', 'w', encoding='utf-8') as file:
                for d in dev:
                    file.write(d + '\n')

            with open(f'{path}/train.txt', 'w', encoding='utf-8') as file:
                for d in train:
                    file.write(d + '\n')

            test_hidden = []
            for t in range(len(test)):
                if random() < hp:
                    test_hidden.append(test[t].hide_aspects(mask="z", mask_size=5))
                else:
                    test_hidden.append(test[t])
            preprocessed_test, _ = preprocess(test_hidden, True, args.lang)

            with open(f'{path}/test.txt', 'w', encoding='utf-8') as file:
                for d in preprocessed_test:
                    file.write(d + '\n')

            pd.to_pickle(labels, f'{path}/test-labels.pk')
            # with open(f'{path}/test-labels.txt', 'w', encoding='utf-8') as file:
            #     for label in labels:
            #         for l in label:
            #             file.write(l)
            #         file.write('\n')

    # for f in range(5):
    #     train = preprocess(np.array(org_reviews)[splits['folds'][str(f)]['train']].tolist(), False, args.lang)
    #     dev = preprocess(np.array(org_reviews)[splits['folds'][str(f)]['valid']].tolist(), False, args.lang)
    #     path = f'{output_path}-fold-{f}'
    #     if not os.path.isdir(path): os.makedirs(path)
    #
    #     with open(f'{path}/dev.txt', 'w') as file:
    #         for d in dev:
    #             file.write(d + '\n')
    #
    #     with open(f'{path}/train.txt', 'w') as file:
    #         for d in train:
    #             file.write(d + '\n')

    # repeat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HAST Wrapper')
    parser.add_argument('--dname', dest='dname', type=str, default='16semeval_rest')
    parser.add_argument('--reviews', dest='reviews', type=str,
                        default='reviews.pkl',
                        help='raw dataset file path')
    parser.add_argument('--splits', dest='splits', type=str,
                        default='splits.json',
                        help='raw dataset file path')
    parser.add_argument('--output', dest='output', type=str, default='data', help='output path')
    parser.add_argument('--lang', dest='lang', type=str, default='eng', help='language')

    args = parser.parse_args()

    for dataset in ['SemEval14L', 'SemEval14R', '2015SB12', '2016SB5']:  # 'SemEval14L', 'SemEval14R', '2015SB12', '2016SB5'
        args.splits = f'data/{dataset}/splits.json'
        for lang in ['eng', 'pes_Arab', 'zho_Hans', 'deu_Latn', 'arb_Arab', 'fra_Latn', 'spa_Latn',
                     'pes_Arab.zho_Hans.deu_Latn.arb_Arab.fra_Latn.spa_Latn']:
            if lang == 'eng':
                args.lang = lang
                args.dname = f'{dataset}'
                args.reviews = f'data/{dataset}/reviews.pkl'
            else:
                args.lang = lang
                args.dname = f'{dataset}-{lang}'
                args.reviews = f'data/{dataset}/reviews.{lang}.pkl'
            print(args)
            main(args)