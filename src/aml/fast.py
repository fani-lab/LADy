import copy
import pickle
import re
from typing import List
import numpy as np
import pandas as pd
import fasttext
import gensim

from .mdl import AbstractAspectModel, AbstractSentimentModel, flatten
from cmn.review import Review

# Utility functions
def add_label_aspect(r):
    r_ = copy.deepcopy(r)
    for i, s in enumerate(r_.sentences):
        for j, _, _ in r.aos[i]: # j is the index of aspect words in sentence s
            for k in j: s[k] = "__label__" + s[k] if s[k].find("__label__") == -1 else s[k]
    return r_

def add_label_sentiment(r):
    r_ = copy.deepcopy(r)
    for i, s in enumerate(r_.sentences):
        for _, _, sentiment in r.aos[i]:
            s.append("__label__" + sentiment)

def add_label(r, label_type):
    if label_type == 'aspect': return add_label_aspect(r)
    elif label_type == 'sentiment': return add_label_sentiment(r)

def review_formatted_file(path, corpus):
    with open(path, 'w', encoding='utf-8') as f:
        for r in corpus: f.write(' '.join(r) + '\n')


class Fast(AbstractAspectModel, AbstractSentimentModel):
    def __init__(self, naspects, nwords): 
        super().__init__(naspects, nwords)
        self.aspect_word_prob = None

    def load(self, path):
        try:
            self.mdl = fasttext.load_model(f'{path}model')
            # assert self.mdl.topics_num_ == self.naspects
            self.dict = pd.read_pickle(f'{path}model.dict')
            self.aspect_word_prob = pd.read_pickle(f'{path}model_aspword_prob.pkl')
        except ValueError:
            raise FileNotFoundError(f'{path}model')
        

    def train(self, reviews_train, reviews_valid, settings, doctype, no_extremes, output):
        corpus, self.dict = self.preprocess(doctype, reviews_train, no_extremes)
        review_formatted_file(f'{output}model.train', corpus)
        self.mdl = fasttext.train_supervised(f'{output}model.train', **settings)
        self.aspect_word_prob = self.generate_aspect_words()

        self.dict.save(f'{output}model.dict')
        self.mdl.save_model(f'{output}model')
        pd.to_pickle(self.aspect_word_prob, f'{output}model_aspword_prob.pkl')
        # do we need cas and perplexity?

    # TODO: see how to integrate this with LADy pipeline
    def infer(self, review: Review, doctype: str):
        return self.mdl.predict(review.get_txt(), k=self.naspects)
    
    @staticmethod
    def preprocess(doctype, reviews, settings=None, label_type='aspect'):
        if not AbstractAspectModel.stop_words:
            import nltk
            AbstractAspectModel.stop_words = nltk.corpus.stopwords.words('english')
    
        reviews_ = []
        if doctype == 'rvw': reviews_ = [np.concatenate(add_label(r, label_type).sentences) for r in reviews]
        elif doctype == 'snt': reviews_ = [s for r in reviews for s in add_label(r, label_type).sentences]
        reviews_ = [[word for word in doc if word not in AbstractAspectModel.stop_words and len(word) > 3 
                     and (re.match('[a-zA-Z]+', word) or re.search('__label__', word))] for doc in reviews_]
        dict = gensim.corpora.Dictionary(reviews_)
        if settings: dict.filter_extremes(no_below=settings['no_below'], no_above=settings['no_above'], keep_n=100000)
        dict.compactify()
        return reviews_, dict
    
    def get_aspect_words(self, aspect, nwords):
        words_prob = []
        print(sorted(self.aspect_word_prob[aspect].items(), key=lambda item: item[1], reverse=True)[:nwords])
        for wp in sorted(self.aspect_word_prob[aspect].items(), key=lambda item: item[1], reverse=True)[:nwords]:
            words_prob.append(wp)
        return words_prob

    def generate_aspect_words(self):
        aw_prob = dict()
        aspects = self.mdl.get_labels()
        n_aspects = len(aspects)
        words = self.mdl.get_words()

        for w in words:
            w_prob = self.mdl.predict(w, k=n_aspects) # the probability of w helping to infer each aspect
            asp = w_prob[0]
            prob = w_prob[1]

            for i, asp in enumerate(asp):
                if asp not in aw_prob: aw_prob[asp] = dict()
                aw_prob[asp][w] = prob[i]
        
        # sort words in each aspect by their probabilities
        for a in aspects:
            aw_prob[a] = {k: v for k, v in sorted(aw_prob[a].items(), key=lambda item: item[1], reverse=True)}

        return aw_prob
    
    def merge_aspects_words(self, r_pred_aspects, nwords):
        # Since predicted aspects are distributions over words, we need to flatten them into list of words.
        # Given a and b are two aspects, we do prob(a) * prob(a_w) for all w \in a and prob(b) * prob(b_w) for all w \in b
        # Then sort.
        result = []
        subr_pred_aspects = r_pred_aspects[0]
        pred_aspects_prob = r_pred_aspects[1]
        subr_pred_aspects_words = []

        for i, a in enumerate(subr_pred_aspects):
            a_p = pred_aspects_prob[i]
            subr_pred_aspects_words.append([(w, a_p * w_p) for w, w_p in self.get_aspect_words(a, nwords)])

        result.append(sorted(flatten(subr_pred_aspects_words), reverse=True, key=lambda t: t[1]))

        return result
    
    def train_sentiment(self, reviews_train, reviews_valid, settings, doctype, no_extremes, output):
        corpus, self.dict = self.preprocess(doctype, reviews_train, no_extremes)
        review_formatted_file(f'{output}model.train', corpus)
        self.mdl = fasttext.train_supervised(f'{output}model.train', **settings, label_type='sentiment')
        self.aspect_word_prob = self.generate_aspect_words()

        self.dict.save(f'{output}model.dict')
        self.mdl.save_model(f'{output}model')
        pd.to_pickle(self.aspect_word_prob, f'{output}model_sword_prob.pkl')
        # do we need cas and perplexity?

    def infer_sentiment(self, review: Review, doctype: str):
        return self.mdl.predict(review.get_txt(), k=self.naspects)