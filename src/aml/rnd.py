import gensim, pickle, random

from .mdl import AbstractAspectModel

class Rnd(AbstractAspectModel):
    def __init__(self, naspects, nwords): super().__init__(naspects, nwords)

    def load(self, path, settings):
        self.dict = gensim.corpora.Dictionary.load(f'{path}model.dict')
        with open(f'{path}model.perf.cas', 'rb') as f: self.cas = pickle.load(f)
        with open(f'{path}model.perf.perplexity', 'rb') as f: self.perplexity = pickle.load(f)

    def infer(self, review, doctype):
        review_ = super(Rnd, self).preprocess(doctype, [review])
        return [[(0, 1)] for r in review_]

    def get_aspect_words(self, aspect_id, nwords): return [(i, 1) for i in random.sample(self.dict.token2id.keys(), min(nwords, len(self.dict)))]

