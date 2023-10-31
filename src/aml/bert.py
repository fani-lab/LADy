from typing import Optional, Tuple, List, TypedDict, Dict
import os, re, random
from argparse import Namespace
import pandas as pd

from aml.mdl import AbstractReviewAnalysisModel, ExtractionCapabilities, flatten
from bert_e2e_absa import work, main as train
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

def write_list_to_file(path: str, data: List[str]) -> None:
    with open(file=path, mode='w', encoding='utf-8') as file:
        for d in data: file.write(d + '\n')


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


def save_train_reviews_to_file(original_reviews: List[Review], output: str) -> List[str]:
    train, _ = convert_reviews_from_lady(original_reviews)

    write_list_to_file(f'{output}/dev.txt', train)
    write_list_to_file(f'{output}/train.txt', train)
    
    return train
    # for h in range(0, 101, 10):
    #     path = f'{output}/latency-{h}'

    #     if not os.path.isdir(path): os.makedirs(path)

    #     with open(f'{path}/dev.txt', 'w', encoding='utf-8') as file:
    #         for d in dev: file.write(d + '\n')

    #     with open(f'{path}/train.txt', 'w', encoding='utf-8') as file:
    #         for d in train: file.write(d + '\n')

def save_test_reviews_to_file(validation_reviews: List[Review], h_ratio: float, output: str) -> None:
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

        write_list_to_file(f'{path}/test.txt', preprocessed_test)

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
class BERT(AbstractReviewAnalysisModel):
    capabilities: ExtractionCapabilities  = ['aspect_detection', 'sentiment_analysis']

    _output_dir_name = 'bert-train' # output dir should contain any train | finetune | fix | overfit
    _data_dir_name   = 'data'

    def __init__(self, naspects, nwords): 
        super().__init__(naspects, nwords, self.capabilities)
    
    def load(self, path):
        path = path[:-1] + f'/{self._data_dir_name}/{self._output_dir_name}/pytorch_model.bin'

        if os.path.isfile(path):
            pass
        else:
            raise FileNotFoundError(f'Model not found for path: {path}')

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
            data_dir = output + f'/{self._data_dir_name}'

            if(not os.path.isdir(data_dir)): os.makedirs(data_dir)

            save_train_reviews_to_file(reviews_train, data_dir)

            args = settings['train']['bert']

            args['data_dir'] = data_dir

            args['output_dir'] = data_dir + f'/{self._output_dir_name}'

            model = train.main(Namespace(**args))

            pd.to_pickle(model, f'{output}.model')


        except Exception as e:
            raise RuntimeError(f'Error in training BERT model: {e}')

    def infer_batch(self, reviews_test, h_ratio, doctype, output):
        output        = f'{output}/{self._data_dir_name}'
        test_data_dir = output + '/tests'
        output_dir    = output + f'/{self._output_dir_name}'

        args = settings['train']['bert']

        save_test_reviews_to_file(reviews_test, settings['test']['h_ratio'], test_data_dir)

        args['output_dir'] = output_dir
        args['absa_home'] = output_dir
        args['ckpt'] = f'{output_dir}/checkpoint-1200'

        pairs = []

        for h in range(0, int(h_ratio * 100 + 1), 10):
            path = f'{test_data_dir}/latency-{h}'

            args['data_dir'] = path 
            result =  work.main(Namespace(**args))

            gold_targets = result['gold_targets']
            unique_predictions = result['unique_predictions']

            pairs.append((flatten(gold_targets), flatten(unique_predictions)))

        return pairs

