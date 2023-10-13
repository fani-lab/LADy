from typing import Tuple, List, TypedDict
from argparse import Namespace
import re
import logging, pickle, pandas as pd, random
import bitermplus as btm, gensim
from mdl import AbstractAspectModel
from bert_e2e_absa import train, work

from src.cmn import review


def convert_review_from_lady(org_reviews: List[review.Review], is_test, lang):
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
        self.cas = pd.read_pickle(f'{path}model.perf.cas')
        self.perplexity = pd.read_pickle(f'{path}model.perf.perplexity')
    
    def train(self,
              reviews_train,
              reviews_valid,
              settings,
              doctype,
              no_extremes,
              output
        ):

        args = {
          'model_type': 'bert',
          'absa_type': absa_type,
          'tfm_mode': 'finetune',
          'fix_tfm': 0,
          'model_name_or_path': 'bert-base-uncased',
          'data_dir': data_dir,
          'task_name': task_name,
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


    def predict(self, absa_home: str, task_name: str, data_dir: str) -> None:
        args = {
            'absa_home': f'{absa_home}',
            'ckpt': f'{absa_home}/checkpoint-1200',
            'model_type': 'bert',
            'data_dir': f'{data_dir}',
            'task_name': f'{task_name}',
            'model_name_or_path': 'bert-base-uncased',
            'cache_dir': 'cache',
            'max_seq_length': 128,
            'tagging_schema': 'BIEOS'
        }

        predict_result = work.main(Namespace(**args))

        unique_predictions = predict_result['unique_predictions']
        gold_targets = predict_result['gold_targets']

    # TODO: I changed the predict method but is there other stuff I need to change?
    def infer_batch(self, reviews_test, h_ratio, doctype) -> None:
      reviews_test_ = []; reviews_aspects = []
      for r in reviews_test:
          r_aspects = [[w for a, o, s in sent for w in a] for sent in r.get_aos()]  # [['service', 'food'], ['service'], ...]
          if len(r_aspects[0]) == 0: continue  # ??
          if random.random() < h_ratio: r_ = r.hide_aspects()
          else: r_ = r
          reviews_aspects.append(r_aspects)
          reviews_test_.append(r_)

      corpus_test, _ = super(BERT, self).preprocess(doctype, reviews_test_)
      corpus_test = [' '.join(doc) for doc in corpus_test]

      reviews_pred_aspects = self.mdl.transform(btm.get_vectorized_docs(corpus_test, self.dict))
      pairs = []
      for i, r_pred_aspects in enumerate(reviews_pred_aspects):
          r_pred_aspects = [[(j, v) for j, v in enumerate(r_pred_aspects)]]
          pairs.extend(list(zip(reviews_aspects[i], self.merge_aspects_words(r_pred_aspects, self.nwords))))

      return pairs


    # TODO: What is this? should I change it?
    def get_aspects_words(self, nwords):
        words: List[str] = []
        probs: List[float] = []

        topic_range_idx = list(range(0, self.naspects))
        top_words = btm.get_top_topic_words(self.mdl, words_num=nwords, topics_idx=topic_range_idx)
        for i in topic_range_idx:
            probs.append(sorted(self.mdl.matrix_topics_words_[i, :]))
            words.append(list(top_words[f'topic{i}']))
        return words, probs

    # TODO: What is this? should I change it?
    def get_aspect_words(self, aspect_id, nwords) -> List[Tuple[str, float]]:
        dict_len = len(self.dict)
        if nwords > dict_len: nwords = dict_len
        topic_range_idx = list(range(0, self.naspects))
        top_words = btm.get_top_topic_words(self.mdl, words_num=nwords, topics_idx=topic_range_idx)
        probs = sorted(self.mdl.matrix_topics_words_[aspect_id, :])
        words = list(top_words[f'topic{aspect_id}'])
        return list(zip(words, probs))
