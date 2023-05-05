import gensim, pickle, random

from .mdl import AbstractAspectModel

class Rnd(AbstractAspectModel):
    def __init__(self, naspects): super().__init__(naspects)

    def load(self, path):
        self.dict = gensim.corpora.Dictionary.load(f'{path}model.dict')
        with open(f'{path}model.perf.cas', 'rb') as f: self.cas = pickle.load(f)
        with open(f'{path}model.perf.perplexity', 'rb') as f: self.perplexity = pickle.load(f)

    def train(self, reviews_train, reviews_valid, settings, doctype, output):
        reviews_ = super().preprocess(doctype, reviews_train)
        self.dict = gensim.corpora.Dictionary(reviews_)
        if settings['no_extremes']: self.dict.filter_extremes(no_below=settings['no_extremes']['no_below'], no_above=settings['no_extremes']['no_above'], keep_n=100000)
        self.dict.compactify()
        # corpus = [self.dict.doc2bow(doc) for doc in reviews_]
        # aspects = self.get_aspects(params.nwords)
        self.cas = 0.00
        self.perplexity = 0.00
        self.dict.save(f'{output}model.dict')

        with open(f'{output}model.perf.cas', 'wb') as f: pickle.dump(self.cas, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{output}model.perf.perplexity', 'wb') as f: pickle.dump(self.perplexity, f, protocol=pickle.HIGHEST_PROTOCOL)

    def infer(self, doctype, review):
        review_ = super().preprocess(doctype, [review])
        return [[(0, 1)] for r in review_]

    def get_aspect(self, topic_id, nwords): return [(i, 1) for i in random.sample(self.dict.token2id.keys(), min(nwords, len(self.dict)))]

