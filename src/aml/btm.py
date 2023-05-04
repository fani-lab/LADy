import logging, pickle, pandas as pd
import bitermplus as btm, gensim

from .mdl import AbstractAspectModel

class Btm(AbstractAspectModel):
    def __init__(self, naspects): super().__init__(naspects)

    def load(self, path):
        self.mdl = pd.read_pickle(f'{path}model')
        assert self.mdl.topics_num_ == self.naspects
        self.dict = pd.read_pickle(f'{path}model.dict')
        with open(f'{path}model.perf.cas', 'rb') as f: self.cas = pickle.load(f)
        with open(f'{path}model.perf.perplexity', 'rb') as f: self.perplexity = pickle.load(f)

    def train(self, reviews_train, reviews_valid, settings, doctype, langaug, output):
        reviews_ = super().preprocess(doctype, reviews_train)
        logging.getLogger().handlers.clear()
        logging.basicConfig(filename=f'{output}model.train.log', format="%(asctime)s:%(levelname)s:%(message)s", level=logging.NOTSET)

        self.dict = gensim.corpora.Dictionary(reviews_)
        if settings['no_extremes']: self.dict.filter_extremes(no_below=settings['no_extremes']['no_below'], no_above=settings['no_extremes']['no_above'], keep_n=100000)
        self.dict.compactify()
        corpus = [' '.join(doc) for doc in reviews_]
        # doc_word_frequency, self.dict, vocab_dict = btm.get_words_freqs(corpus)
        doc_word_frequency, self.dict, vocab_dict = btm.get_words_freqs(corpus, **{'vocabulary': self.dict.token2id})
        docs_vec = btm.get_vectorized_docs(corpus, self.dict)
        biterms = btm.get_biterms(docs_vec)

        self.mdl = btm.BTM(doc_word_frequency, self.dict, seed=settings['seed'], T=self.naspects, M=settings['nwords'], alpha=1.0/self.naspects, beta=0.01) #https://bitermplus.readthedocs.io/en/latest/bitermplus.html#bitermplus.BTM
        self.mdl.fit(biterms, iterations=settings['iter'], verbose=True)

        self.cas = self.mdl.coherence_
        self.perplexity = 0#self.mdl.perplexity_=> Process finished with exit code -1073741819 (0xC0000005)
        pd.to_pickle(self.dict, f'{output}model.dict')
        pd.to_pickle(self.mdl, f'{output}model')
        with open(f'{output}model.perf.cas', 'wb') as f: pickle.dump(self.cas, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{output}model.perf.perplexity', 'wb') as f: pickle.dump(self.perplexity, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_aspects(self, nwords):
        words = []; probs = []
        topic_range_idx = list(range(0, self.naspects))
        top_words = btm.get_top_topic_words(self.mdl, words_num=nwords, topics_idx=topic_range_idx)
        for i in topic_range_idx:
            probs.append(sorted(self.mdl.matrix_topics_words_[i, :]))
            words.append(list(top_words[f'topic{i}']))
        return words, probs

    def get_aspect(self, topic_id, nwords):
        dict_len = len(self.dict)
        if nwords > dict_len: nwords = dict_len
        topic_range_idx = list(range(0, self.naspects))
        top_words = btm.get_top_topic_words(self.mdl, words_num=nwords, topics_idx=topic_range_idx)
        probs = sorted(self.mdl.matrix_topics_words_[topic_id, :])
        words = list(top_words[f'topic{topic_id}'])
        return list(zip(words, probs))

    def infer(self, doctype, review):
        review_aspects = []
        review_ = super().preprocess(doctype, [review])
        for r in review_: review_aspects.append([(i, p) for i, p in enumerate(self.mdl.transform(btm.get_vectorized_docs([' '.join(r)], self.dict))[0])])
        return review_aspects

