import logging, pickle
import pandas as pd
import nltk
stop_words = nltk.corpus.stopwords.words('english')
import bitermplus as btm
import gensim
import numpy as np

import params
from .mdl import AbstractAspectModel


class Btm(AbstractAspectModel):
    def __init__(self, reviews, naspects, no_extremes, output):
        super().__init__(reviews, naspects, no_extremes, output)

    def load(self):
        self.mdl = pd.read_pickle(f'{self.path}model.pkl')
        assert self.mdl.topics_num_ == self.naspects
        self.dict = pd.read_pickle(f'{self.path}model.dict.pkl')
        with open(f'{self.path}model.perf.cas', 'rb') as f: self.cas = pickle.load(f)
        with open(f'{self.path}model.perf.perplexity', 'rb') as f: self.perplexity = pickle.load(f)

    def train(self, doctype, cores, iter, seed):
        reviews_ = super().preprocess(doctype, self.reviews)

        logging.basicConfig(filename=f'{self.path}model.train.log', format="%(asctime)s:%(levelname)s:%(message)s", level=logging.NOTSET)

        # reviews_ = [' '.join(text) for text in reviews_]
        # doc_word_frequency, self.dict, vocab_dict = btm.get_words_freqs(r, max_df=self.no_extremes['no_above'])

        self.dict = gensim.corpora.Dictionary(reviews_)
        if self.no_extremes:
            self.dict.filter_extremes(no_below=self.no_extremes['no_below'], no_above=self.no_extremes['no_above'],
                                      keep_n=100000)
        self.dict.compactify()

        tfidf_model = gensim.models.TfidfModel([self.dict.doc2bow(text) for text in reviews_], id2word=self.dict,
                                               normalize=True)
        tfidf = tfidf_model[[self.dict.doc2bow(text) for text in reviews_]]
        doc_word_frequency = gensim.matutils.corpus2csc(tfidf, num_terms=len(self.dict)).T
        self.dict = np.asarray(list(self.dict.token2id.keys()))
        doc_word_frequency[doc_word_frequency > 0] = 1
        reviews_ = [' '.join(text) for text in reviews_]

        docs_vec = btm.get_vectorized_docs(reviews_, self.dict)
        biterms = btm.get_biterms(docs_vec)
        self.mdl = btm.BTM(doc_word_frequency, self.dict, seed=params.seed, T=self.naspects, M=params.nwords, alpha=50/self.naspects, beta=0.01) #https://bitermplus.readthedocs.io/en/latest/bitermplus.html#bitermplus.BTM
        self.mdl.fit(biterms, iterations=params.iter_c, verbose=False)

        self.cas = self.mdl.coherence_
        self.perplexity = self.mdl.perplexity_
        aspects, probs = self.get_aspects(params.nwords)
        pd.to_pickle(self.dict, f'{self.path}model.dict.pkl')
        pd.to_pickle(self.mdl, f'{self.path}model.pkl')
        with open(f'{self.path}model.perf.cas', 'wb') as f: pickle.dump(self.cas, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{self.path}model.perf.perplexity', 'wb') as f: pickle.dump(self.perplexity, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_aspects(self, nwords):
        words = []
        probs = []
        topic_range_idx = list(range(0, self.naspects))
        top_words = btm.get_top_topic_words(self.mdl, words_num=nwords, topics_idx=topic_range_idx)
        for i in topic_range_idx:
            probs.append(sorted(self.mdl.matrix_topics_words_[i, :]))
            words.append(list(top_words[f'topic{i}']))
        return words, probs

    def show_topic(self, topic_id, nwords):
        dict_len = len(self.dict)
        if nwords > dict_len:
            nwords = dict_len
        topic_range_idx = list(range(0, self.naspects))
        top_words = btm.get_top_topic_words(self.mdl, words_num=nwords, topics_idx=topic_range_idx)
        probs = sorted(self.mdl.matrix_topics_words_[topic_id, :])
        words = list(top_words[f'topic{topic_id}'])
        return list(zip(words, probs))
    def infer(self, doctype, review):
        review_aspects = []
        review_ = super().preprocess(doctype, [review])
        t_t = 'Text'
        for r in review_:
            review_aspects.append([(i, p) for i, p in enumerate(self.mdl.transform(btm.get_vectorized_docs([' '.join(r)], self.dict))[0])])
        return review_aspects

