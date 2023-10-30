import copy
import re
from typing import List
import numpy as np
import pandas as pd
import fasttext
import gensim

from .mdl import AbstractAspectModel
from cmn.review import Review

# Utility functions
def add_label(r):
    r_ = copy.deepcopy(r)
    for i, s in enumerate(r_.sentences):
        for j, _, _ in r.aos[i]: # j is the index of aspect words in sentence s
            for k in j: s[k] = "__label__" + s[k] if s[k].find("__label__") == -1 else s[k]
    return r_

def review_formatted_file(path, corpus):
    with open(path, 'w') as f:
        for r in corpus: f.write(' '.join(r) + '\n')


class Fast(AbstractAspectModel):
    def __init__(self, naspects, nwords): super().__init__(naspects, nwords)

    def load(self, path):
        self.mdl = fasttext.load_model(f'{path}model')
        # assert self.mdl.topics_num_ == self.naspects
        # TODO: see how to incorporate naspects
        self.dict = pd.read_pickle(f'{path}model.dict')

    def train(self, reviews_train, reviews_valid, settings, doctype, no_extremes, output):
        corpus, self.dict = self.preprocess(doctype, reviews_train, no_extremes)
        review_formatted_file(f'{output}model.train', corpus)
        self.mdl = fasttext.train_supervised(f'{output}model.train', **settings)

        self.dict.save(f'{output}model.dict')
        self.mdl.save_model(f'{output}model')
        # do we need cas and perplexity?

    # TODO: see how to integrate this with LADy pipeline
    def infer(self, review: Review, doctype: str):
        return self.mdl.predict(review.get_txt(), k=self.naspects)
    
    @staticmethod
    def preprocess(doctype, reviews, settings=None):
        if not AbstractAspectModel.stop_words:
            import nltk
            AbstractAspectModel.stop_words = nltk.corpus.stopwords.words('english')
    
        reviews_ = []
        if doctype == 'rvw': reviews_ = [np.concatenate(add_label(r).sentences) for r in reviews]
        elif doctype == 'snt': reviews_ = [s for r in reviews for s in add_label(r).sentences]
        reviews_ = [[word for word in doc if word not in AbstractAspectModel.stop_words and len(word) > 3 
                     and (re.match('[a-zA-Z]+', word) or re.search('__label__', word))] for doc in reviews_]
        dict = gensim.corpora.Dictionary(reviews_)
        if settings: dict.filter_extremes(no_below=settings['no_below'], no_above=settings['no_above'], keep_n=100000)
        dict.compactify()
        return reviews_, dict