import gensim, pickle, random

import nltk
stop_words = nltk.corpus.stopwords.words('english')


from .mdl import AbstractAspectModel


class Rnd(AbstractAspectModel):
    def __init__(self, reviews, naspects, no_extremes, output):
        super().__init__(reviews, naspects, no_extremes, output)

    def load(self):
        # num_topics = 25
        self.dict = gensim.corpora.Dictionary.load(f'{self.path}model.dict')
        with open(f'{self.path}model.perf.cas', 'rb') as f: self.cas = pickle.load(f)
        with open(f'{self.path}model.perf.perplexity', 'rb') as f: self.perplexity = pickle.load(f)

    def train(self, doctype, cores, iter, seed):
        reviews_ = super().preprocess(doctype, self.reviews)
        self.dict = gensim.corpora.Dictionary(reviews_)
        if self.no_extremes: self.dict.filter_extremes(no_below=self.no_extremes['no_below'], no_above=self.no_extremes['no_above'], keep_n=100000)
        self.dict.compactify()
        # corpus = [self.dict.doc2bow(doc) for doc in reviews_]
        # aspects = self.get_aspects(params.nwords)
        self.cas = 0.00
        self.perplexity = 0.00
        self.dict.save(f'{self.path}model.dict')
        with open(f'{self.path}model.perf.cas', 'wb') as f: pickle.dump(self.cas, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{self.path}model.perf.perplexity', 'wb') as f: pickle.dump(self.perplexity, f, protocol=pickle.HIGHEST_PROTOCOL)

    def infer(self, doctype, review):
        review_aspects = list(self.dict.token2id.keys())
        random.shuffle(review_aspects)
        return review_aspects


