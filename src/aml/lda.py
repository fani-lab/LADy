import gensim, logging, pickle, re
import pandas as pd
import numpy as np
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
from gensim.models.callbacks import PerplexityMetric, ConvergenceMetric, CoherenceMetric
from gensim.models.coherencemodel import CoherenceModel
import nltk
stop_words = nltk.corpus.stopwords.words('english')


from src import params
from .mdl import AbstractAspectModel

class Lda(AbstractAspectModel):
    def __init__(self, reviews, naspects, no_extremes, output):
        super().__init__(reviews, naspects, no_extremes, output)

    def load(self):
        self.mdl = gensim.models.LdaModel.load(f'{self.path}model')
        assert self.mdl.num_topics == self.naspects
        self.dict = gensim.corpora.Dictionary.load(f'{self.path}model.dict')
        with open(f'{self.path}model.perf.cas', 'rb') as f: self.cas = pickle.load(f)
        with open(f'{self.path}model.perf.perplexity', 'rb') as f: self.perplexity = pickle.load(f)

    def train(self, doctype, cores, iter, seed):
        reviews_ = super().preprocess(doctype, self.reviews)
        self.dict = gensim.corpora.Dictionary(reviews_)
        if self.no_extremes: self.dict.filter_extremes(no_below=self.no_extremes['no_below'], no_above=self.no_extremes['no_above'], keep_n=100000)
        self.dict.compactify()
        corpus = [self.dict.doc2bow(doc) for doc in reviews_]

        logging.basicConfig(filename=f'{self.path}model.train.log', format="%(asctime)s:%(levelname)s:%(message)s", level=logging.NOTSET)
        # callback functions cannot be applied in parallel lda, only for LdaModel()
        # perplexity_logger = PerplexityMetric(corpus=corpus, logger='shell')
        # convergence_logger = ConvergenceMetric(logger='shell')
        # coherence_cv_logger = CoherenceMetric(corpus=corpus, logger='shell', coherence='c_v', texts=reviews_)
        # self.model = gensim.models.wrappers.LdaMallet(mallet, corpus, num_topics=self.naspects, id2word=self.dict, workers=cores, iterations=iter, callback=)
        self.mdl = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics=self.naspects, id2word=self.dict, workers=cores, passes=iter, random_state=seed, per_word_topics=True)

        # TODO: quality diagram ==> https://www.meganstodel.com/posts/callbacks/
        aspects, probs = self.get_aspects(params.nwords)
        # https://stackoverflow.com/questions/50607378/negative-values-evaluate-gensim-lda-with-topic-coherence
        # umass: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://aclanthology.org/D11-1024.pdf
        # [-inf, 0]: close to zero, the better
        self.cas = CoherenceModel(model=self.mdl, topics=aspects, corpus=corpus, dictionary=self.dict, coherence='c_v', texts=reviews_).get_coherence_per_topic()
        self.perplexity = self.mdl.log_perplexity(corpus)
        self.dict.save(f'{self.path}model.dict')
        self.mdl.save(f'{self.path}model')
        with open(f'{self.path}model.perf.cas', 'wb') as f: pickle.dump(self.cas, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{self.path}model.perf.perplexity', 'wb') as f: pickle.dump(self.perplexity, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_aspects(self, nwords):
        print("*")
        # self.model.get_topics() does not have words
        # self.model.show_topics() and model.show_topic()
        words = []; probs = []
        print("print_topics:/t", self.mdl.print_topics(-1, num_words=nwords))
        for idx, aspect in self.mdl.print_topics(-1, num_words=nwords):
            words.append([]); probs.append([])
            words_probs = aspect.split('+')
            for word_prob in words_probs:
                probs[-1].append(word_prob.split('*')[0])
                words[-1].append(word_prob.split('*')[1].split('"')[1])
        print("words", words)
        return words, probs

    def infer(self, doctype, review):
        review_aspects = []
        review_ = super().preprocess(doctype, [review])
        for r in review_: review_aspects.append(self.mdl.get_document_topics(self.dict.doc2bow(r), minimum_probability=self.mdl.minimum_probability))
        return review_aspects


