import gensim, logging, pickle
import re
import pandas as pd
import numpy as np
from octis.models.NeuralLDA import NeuralLDA
from octis.models.CTM import CTM
from octis.models.ETM import ETM
from octis.dataset.dataset import Dataset
import os
import string
from octis.preprocessing.preprocessing import Preprocessing
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

from sklearn.feature_extraction.text import CountVectorizer

import gensim
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora


import nltk

stop_words = nltk.corpus.stopwords.words('english')

import params
from .mdl import AbstractAspectModel


class Neural(AbstractAspectModel):
    def __init__(self, reviews, naspects, no_extremes, output):
        super().__init__(reviews, naspects, no_extremes, output)

    def load(self):
        self.mdl = pd.read_pickle(f'{self.path}model.pkl')
        # with open('model.dict') as f:
        #     self.dict = f.read().splitlines()
        self.naspects = pd.read_pickle(f'{self.path}naspects.pkl')
        self.dict = pd.read_pickle(f'{self.path}model.dict.pkl')
        with open(f'{self.path}model.perf.cas', 'rb') as f: self.cas = pickle.load(f)
        with open(f'{self.path}model.perf.perplexity', 'rb') as f: self.perplexity = pickle.load(f)
    # def preprocess(doctype, reviews):
    #     reviews_ = [s for r in reviews for s in r.sentences]

    def preprocess(self, doctype, reviews):
        removed_indices = []
        if doctype == 'rvw': reviews_ = [np.concatenate(r.sentences) for r in reviews]
        else:
            for i, r in enumerate(reviews):
                if not r.sentences:
                    removed_indices.append(i)
            reviews_ = [s for r in reviews for s in r.sentences]
        return [[word for word in doc if word not in stop_words and len(word) > 3 and re.match('[a-zA-Z]+', word)] for doc in reviews_], removed_indices

    def train(self, doctype, cores, iter, seed, test=None):

        model = ETM(num_topics=self.naspects, batch_size=params.iter_c)
        reviews_ = super().preprocess(doctype, self.reviews)
        reviews_ = [' '.join(text) for text in reviews_]
        train_tag = ['train' for r in reviews_]

        test_reviews_, self.removed_indices = self.preprocess(doctype, test)
        test = [' '.join(text) for text in test_reviews_]
        test_tag = ['test' for r in test]
        vectorizer = CountVectorizer()
        vectorizer.fit_transform(reviews_)
        # Get the list of unique words
        self.dict = vectorizer.get_feature_names()

        model_path = self.path[:self.path.rfind("/")]
        with open(f'{model_path}/vocabulary.txt', "w", encoding="utf-8") as file:
            for item in self.dict:
                file.write("%s\n" % item)
        with open(f'{model_path}/corpus.tsv', "w", encoding="utf-8") as outfile:
            for i in range(len(reviews_)):
                if reviews_[i] == '':
                    continue
                outfile.write("{}\t{}\n".format(reviews_[i], train_tag[i]))
                # if i < len(reviews_)-100:
                #     outfile.write("{}\t{}\n".format(reviews_[i], train_tag[i]))
                # else:
                #     outfile.write("{}\t{}\n".format(reviews_[i], 'val'))
            for i in range(len(test)):
                outfile.write("{}\t{}\n".format(test[i], test_tag[i]))

        dataset = Dataset()
        dataset.load_custom_dataset_from_folder(f'{model_path}')

        # Dataset.save(dataset, f'{self.path}model.dataset')
        # Dataset._save_vocabulary(dataset, f'{self.path}model.dict')

        self.dict = dataset.get_vocabulary()
        self.mdl = model.train_model(dataset)
        # npmi = Coherence(texts=dataset.get_corpus(), topk=params.nwords, measure='u_mass')
        npmi = Coherence(texts=dataset.get_corpus(), measure='u_mass')
        self.cas = npmi.score(self.mdl)
        self.perplexity = 0

        pd.to_pickle(self.dict, f'{self.path}model.dict.pkl')
        pd.to_pickle(self.removed_indices, f'{self.path}ridx.pkl')
        pd.to_pickle(self.mdl, f'{self.path}model.pkl')
        pd.to_pickle(self.naspects, f'{self.path}naspects.pkl')
        with open(f'{self.path}model.perf.cas', 'wb') as f:
            pickle.dump(self.cas, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{self.path}model.perf.perplexity', 'wb') as f:
            pickle.dump(self.perplexity, f, protocol=pickle.HIGHEST_PROTOCOL)

    def show_topic(self, topic_id, nwords):
        word_list = self.mdl['topics'][topic_id]
        probs = []
        matrix = self.mdl['topic-word-matrix'][topic_id]
        for w in word_list:
            probs.append(matrix[self.dict.index(w)])
        return list(zip(word_list, probs))

    def infer(self, doctype, review, idx=None):
        review_aspects = []
        for t in range(self.naspects):
            review_aspects.append((t, self.mdl['test-topic-document-matrix'].T[idx][t]))
        return [review_aspects]
