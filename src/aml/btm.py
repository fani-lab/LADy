import logging, pickle, pandas as pd, random
import bitermplus as btm, gensim

from .mdl import AbstractAspectModel

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
class Btm(AbstractAspectModel):
    def __init__(self, naspects, nwords): super().__init__(naspects, nwords)

    def load(self, path):
        self.mdl = pd.read_pickle(f'{path}model')
        assert self.mdl.topics_num_ == self.naspects
        self.dict = pd.read_pickle(f'{path}model.dict')
        self.cas = pd.read_pickle(f'{path}model.perf.cas')
        self.perplexity = pd.read_pickle(f'{path}model.perf.perplexity')

    def train(self, reviews_train, reviews_valid, settings, doctype, no_extremes, output):
        corpus, self.dict = super(Btm, self).preprocess(doctype, reviews_train, no_extremes)
        corpus = [' '.join(doc) for doc in corpus]

        logging.getLogger().handlers.clear()
        logging.basicConfig(filename=f'{output}model.train.log', format="%(asctime)s:%(levelname)s:%(message)s", level=logging.NOTSET)
        # doc_word_frequency, self.dict, vocab_dict = btm.get_words_freqs(corpus)
        doc_word_frequency, self.dict, vocab_dict = btm.get_words_freqs(corpus, **{'vocabulary': self.dict.token2id})
        docs_vec = btm.get_vectorized_docs(corpus, self.dict)
        biterms = btm.get_biterms(docs_vec)

        self.mdl = btm.BTM(doc_word_frequency, self.dict, T=self.naspects, M=self.nwords, alpha=1.0/self.naspects, seed=settings['seed'], beta=0.01) #https://bitermplus.readthedocs.io/en/latest/bitermplus.html#bitermplus.BTM
        self.mdl.fit(biterms, iterations=settings['iter'], verbose=True)

        self.cas = self.mdl.coherence_
        self.perplexity_ = self.mdl.perplexity_ ##DEBUG: Process finished with exit code -1073741819 (0xC0000005)
        pd.to_pickle(self.dict, f'{output}model.dict')
        pd.to_pickle(self.mdl, f'{output}model')
        pd.to_pickle(self.cas, f'{output}model.perf.cas')
        pd.to_pickle(self.perplexity, f'{output}model.perf.perplexity')

    def get_aspects_words(self, nwords):
        words = []; probs = []
        topic_range_idx = list(range(0, self.naspects))
        top_words = btm.get_top_topic_words(self.mdl, words_num=nwords, topics_idx=topic_range_idx)
        for i in topic_range_idx:
            probs.append(sorted(self.mdl.matrix_topics_words_[i, :]))
            words.append(list(top_words[f'topic{i}']))
        return words, probs

    def get_aspect_words(self, aspect_id, nwords):
        dict_len = len(self.dict)
        if nwords > dict_len: nwords = dict_len
        topic_range_idx = list(range(0, self.naspects))
        top_words = btm.get_top_topic_words(self.mdl, words_num=nwords, topics_idx=topic_range_idx)
        probs = sorted(self.mdl.matrix_topics_words_[aspect_id, :])
        words = list(top_words[f'topic{aspect_id}'])
        return list(zip(words, probs))

    def infer_batch(self, reviews_test, h_ratio, doctype, output):
        reviews_test_ = []; reviews_aspects = []
        for r in reviews_test:
            r_aspects = [[w for a, o, s in sent for w in a] for sent in r.get_aos()]  # [['service', 'food'], ['service'], ...]
            if len(r_aspects[0]) == 0: continue  # ??
            if random.random() < h_ratio: r_ = r.hide_aspects()
            else: r_ = r
            reviews_aspects.append(r_aspects)
            reviews_test_.append(r_)

        corpus_test, _ = super(Btm, self).preprocess(doctype, reviews_test_)
        corpus_test = [' '.join(doc) for doc in corpus_test]

        reviews_pred_aspects = self.mdl.transform(btm.get_vectorized_docs(corpus_test, self.dict))
        pairs = []
        for i, r_pred_aspects in enumerate(reviews_pred_aspects):
            r_pred_aspects = [[(j, v) for j, v in enumerate(r_pred_aspects)]]
            pairs.extend(list(zip(reviews_aspects[i], self.merge_aspects_words(r_pred_aspects, self.nwords))))

        return pairs

