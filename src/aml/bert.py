from typing import Optional, Tuple, List, TypedDict, Dict
from itertools import chain 
import os, re, random
from argparse import Namespace
import pandas as pd

from aml.mdl import AbstractAspectModel
from bert_e2e_absa import train, work
from cmn.review import Review
from params import settings

#--------------------------------------------------------------------------------------------------
# Typings
#--------------------------------------------------------------------------------------------------
class FoldItem(TypedDict):
    train: List[int]
    valid: List[int]

class Split(TypedDict):
    folds: Dict[str, FoldItem]
    test: List[int]

#--------------------------------------------------------------------------------------------------
# Utilities
#--------------------------------------------------------------------------------------------------
def convert_reviews_from_lady(original_reviews: List[Review]) -> Tuple[List[str], List[List[str]]]:
    reviews_list = []
    label_list = []

    # Due to model cannot handle long sentences, we need to filter out long sentences
    REVIEW_MAX_LENGTH = 511

    for r in original_reviews:
        if not len(r.aos[0]): continue
        else:
            aspect_ids = []

            for aos_instance in r.aos[0]: aspect_ids.extend(aos_instance[0])

            text = re.sub(r'\s{2,}', ' ', ' '.join(r.sentences[0]).strip()) + '####'

            for idx, word in enumerate(r.sentences[0]):
                if idx in aspect_ids:
                    tag = word + '=T-POS' + ' '
                    text += tag
                else:
                    tag = word + '=O' + ' '
                    text += tag

            if len(text.rstrip()) > REVIEW_MAX_LENGTH: continue

            reviews_list.append(text.rstrip())

            aos_list_per_review = []

            for idx, word in enumerate(r.sentences[0]):
                if idx in aspect_ids: aos_list_per_review.append(word)

            label_list.append(aos_list_per_review)

    return reviews_list, label_list


def save_train_reviews_to_file(original_reviews: List[Review], output: str) -> None:
    train, _ = convert_reviews_from_lady(original_reviews)
    dev, _ = convert_reviews_from_lady(original_reviews)

    for h in range(0, 101, 10):
        output = f'{output}/latency-{h}'

        if not os.path.isdir(output): os.makedirs(output)

        with open(f'{output}/dev.txt', 'w', encoding='utf-8') as file:
            for d in dev: file.write(d + '\n')

        with open(f'{output}/train.txt', 'w', encoding='utf-8') as file:
            for d in train: file.write(d + '\n')

def save_test_reviews_to_file(validation_reviews: List[Review], h_ratio: float, output: str) -> None:
    if not os.path.isdir(output): os.makedirs(output)
    
    _, labels = convert_reviews_from_lady(validation_reviews)

    for h in range(0, int(h_ratio * 100 + 1), 10):
        hp = h / 100

        path = f'{output}/latency-{h}'

        if not os.path.isdir(path): os.makedirs(path)

        test_hidden = []

        for t in range(len(validation_reviews)):
            if random.random() < hp:
                test_hidden.append(validation_reviews[t].hide_aspects(mask='z', mask_size=5))
            else: test_hidden.append(validation_reviews[t])

        preprocessed_test, _ = convert_reviews_from_lady(test_hidden)

        with open(f'{path}/test.txt', 'w', encoding='utf-8') as file:
            for d in preprocessed_test: file.write(d + '\n')

        pd.to_pickle(labels, f'{path}/test-labels.pk')
#--------------------------------------------------------------------------------------------------
# Class Definition
#--------------------------------------------------------------------------------------------------

# TODO: Change these
# @inproceedings{DBLP:conf/www/YanGLC13,
#   author       = {Xiaohui Yan and Jiafeng Guo and Yanyan Lan and Xueqi Cheng},
#   title        = {A biterm topic model for short texts},
#   booktitle    = {22nd International World Wide Web Conference, {WWW} '13, Rio de Janeiro, Brazil, May 13-17, 2013},
#   pages        = {1445--1456},
#   publisher    = {International World Wide Web Conferences Steering Committee / {ACM}},
#   year         = {2013},
#   url          = {https://doi.org/10.1145/2488388.2488514},
#   biburl       = {https://dblp.org/rec/conf/www/YanGLC13.bib},
# }
class BERT(AbstractAspectModel):
    def __init__(self, naspects: int, nwords: int): 
        super().__init__(naspects, nwords)
    
    # TODO: Change this
    def load(self, path):
        raise FileNotFoundError('')

    def train(self,
              reviews_train: List[Review],
              reviews_validation: Optional[List[Review]],
              am: str,
              doctype: Optional[str],
              no_extremes: Optional[bool],
              output: str
    ):
        try:
            output = output[:-1]
            cache_dir = f'{output}/cache'

            if(not os.path.isdir(output)): os.makedirs(output)

            save_train_reviews_to_file(reviews_train, output)

            if (reviews_validation is None) or (len(reviews_validation) == 0): raise Exception('Validation set is empty')

            save_test_reviews_to_file(reviews_validation, settings['test']['h_ratio'],output)

            args = settings['train']['bert']

            for h in range(0, 101, 10):
                output_ = f'{output}/latency-{h}'

                args['data_dir'] = output_
                args['output_dir'] = output_

                model = train.main(Namespace(**args))

                pd.to_pickle(model, f'{output_}model')


        except Exception as e:
            raise RuntimeError(f'Error in training BERT model: {e}')

    def infer_batch(self, reviews_test, h_ratio, doctype, output):
        args = settings['train']['bert']

        args['data_dir'] = output
        args['output_dir'] = output
        args['absa_home'] = output
        args['ckpt'] = f'{output}checkpoint-1200'

        pairs =  work.main(Namespace(**args))

        gold_targets = pairs['gold_targets']
        unique_predictions = pairs['unique_predictions']
        flattened_unique_predictions = list(chain(*unique_predictions))

        return (gold_targets, flattened_unique_predictions)

