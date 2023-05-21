import gensim, pandas as pd, random

from .mdl import AbstractAspectModel

class Rnd(AbstractAspectModel):
    def __init__(self, naspects, nwords): super().__init__(naspects, nwords)

    def load(self, path):
        self.dict = gensim.corpora.Dictionary.load(f'{path}model.dict')
        pd.to_pickle(self.cas, f'{path}model.perf.cas')
        pd.to_pickle(self.perplexity, f'{path}model.perf.perplexity')

    def infer(self, review, doctype):
        review_ = super(Rnd, self).preprocess(doctype, [review])
        return [[(0, 1)] for r in review_]

    def get_aspect_words(self, aspect_id, nwords): return [(i, 1) for i in random.sample(self.dict.token2id.keys(), min(nwords, len(self.dict)))]

