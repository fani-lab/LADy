from typing import Optional, Tuple, List, TypedDict, Dict
import os
import re
import random
from argparse import Namespace
import pandas as pd
from aml.mdl import AbstractAspectModel
from bert_e2e_absa import train, work

from cmn.review import Review

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
        if not len(r.aos[0]):
            continue
        else:
            aos_list = []

            for aos_instance in r.aos[0]:
                aos_list.extend(aos_instance[0])

            text = re.sub(r'\s{2,}', ' ', ' '.join(r.sentences[0]).strip()) + '####'

            for idx, word in enumerate(r.sentences[0]):
                if idx in aos_list:
                    tag = word + '=T-POS' + ' '
                    text += tag
                else:
                    tag = word + '=O' + ' '
                    text += tag

            if len(text.rstrip()) > REVIEW_MAX_LENGTH:
                continue

            reviews_list.append(text.rstrip())

            aos_list_per_review = []

            for idx, word in enumerate(r.sentences[0]):
                if idx in aos_list:
                    aos_list_per_review.append(word)

            label_list.append(aos_list_per_review)

    return reviews_list, label_list


def save_train_reviews_to_file(original_reviews: List[Review], output: str) -> None:
    train, _ = convert_reviews_from_lady(original_reviews)
    dev, _ = convert_reviews_from_lady(original_reviews)

    if not os.path.isdir(output): 
        os.makedirs(output)

    path_witout_end_dot = output[:-1]
    os.mkdir(path_witout_end_dot)

    for h in range(0, 101, 10):
        output = f'{path_witout_end_dot}/latency-{h}'

        os.mkdir(output)

        with open(f'{output}/dev.txt', 'w', encoding='utf-8') as file:
            for d in dev:
                file.write(d + '\n')

        with open(f'{output}/train.txt', 'w', encoding='utf-8') as file:
            for d in train:
                file.write(d + '\n')

def save_test_reviews_to_file(validation_reviews: List[Review], h_ratio: int, output: str) -> None:
    path_witout_end_dot = output[:-1]

    if not os.path.isdir(output):
        raise Exception(f'Output path {output} does not exist')
    
    _, labels = convert_reviews_from_lady(validation_reviews)


    for h in range(0, h_ratio * 100 + 1, 10):
        hp = h / 100

        path = f'{path_witout_end_dot}/latency-{h}'

        if not os.path.isdir(path):
            os.makedirs(path)

        test_hidden = []
        for t in range(len(validation_reviews)):
            if random.random() < hp:
                test_hidden.append(validation_reviews[t].hide_aspects(mask='z', mask_size=5))
            else:
                test_hidden.append(validation_reviews[t])
        preprocessed_test, _ = convert_reviews_from_lady(test_hidden)

        with open(f'{path}/test.txt', 'w', encoding='utf-8') as file:
            for d in preprocessed_test:
                file.write(d + '\n')

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
    def __init__(self, naspects, nwords): 
        super().__init__(naspects, nwords)

    # TODO: What is this? should I change it?
    def load(self, path):
        self.mdl = pd.read_pickle(f'{path}model')
        assert self.mdl.topics_num_ == self.naspects
        self.dict = pd.read_pickle(f'{path}model.dict')
    
    def train(self,
              reviews_train: List[Review],
              reviews_valid: Optional[List[Review]],
              am: str,
              doctype: Optional[str],
              no_extremes: Optional[bool],
              output: str
    ):
        try:
            save_train_reviews_to_file(reviews_train, output)

            args = {
            'model_type': 'bert',
            'absa_type': 'tfm',
            'tfm_mode': 'finetune',
            'fix_tfm': 0,
            'model_name_or_path': 'bert-base-uncased',
            'data_dir': output,
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
            'max_steps': 1500
            }

            model = train.main(Namespace(**args))

            pd.to_pickle(model, f'{output}model')
        except Exception as e:
            raise RuntimeError(f'Error in training BERT model: {e}')

    def infer_batch(self, reviews_test, h_ratio, doctype, output):
        save_test_reviews_to_file(reviews_test, h_ratio, output)

        absa_home = 'CHANGE_ME'
        args = {
            # TODO: change this to the model directory
            'absa_home': f'{absa_home}',
            'ckpt': f'{absa_home}/checkpoint-1200',
            'model_type': 'bert',
            'data_dir': f'{output}',
            'task_name': 'lady',
            'model_name_or_path': 'bert-base-uncased',
            'cache_dir': 'cache',
            'max_seq_length': 128,
            'tagging_schema': 'BIEOS'
        }

        return work.main(Namespace(**args))
