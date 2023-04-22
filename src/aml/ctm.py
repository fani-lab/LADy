import pickle
import numpy as np
import pandas as pd
import torch
import random
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
from contextualized_topic_models.evaluation.measures import CoherenceUMASS

import nltk
from nltk.corpus import stopwords as stop_words

# nltk.download('stopwords')

import params
from .mdl import AbstractAspectModel


class CTM(AbstractAspectModel):
    def __init__(self, reviews, naspects, no_extremes, output):
        super().__init__(reviews, naspects, no_extremes, output)

    def load(self):
        self.tp = pd.read_pickle(f'{self.path}model.tp.pkl')
        self.mdl = CombinedTM(bow_size=len(self.tp.vocab), contextual_size=768, n_components=self.naspects,
                              num_epochs=100)
        self.model_path = pd.read_pickle(f'{self.path}model.path.pkl')
        self.mdl.load(f'{self.path[:-1]}/{self.model_path}', epoch=99)
        # self.mdl.model.load_state_dict(torch.load(f'{self.path}model.pth'))
        self.dict = pd.read_pickle(f'{self.path}model.dict.pkl')
        with open(f'{self.path}model.perf.cas', 'rb') as f: self.cas = pickle.load(f)
        with open(f'{self.path}model.perf.perplexity', 'rb') as f: self.perplexity = pickle.load(f)

        torch.manual_seed(params.seed)
        torch.cuda.manual_seed(params.seed)
        np.random.seed(params.seed)
        random.seed(params.seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True

    def train(self, doctype, cores, iter, seed):

        torch.manual_seed(params.seed)
        torch.cuda.manual_seed(params.seed)
        np.random.seed(params.seed)
        random.seed(params.seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True

        epoch = 100
        preprocessed_documents, unpreprocessed_corpus, vocab, retained_indices = self.preprocess(doctype, self.reviews)
        self.tp = TopicModelDataPreparation("all-mpnet-base-v2")

        training_dataset = self.tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)
        self.dict = self.tp.vocab
        self.mdl = CombinedTM(bow_size=len(self.tp.vocab), contextual_size=768, n_components=self.naspects,
                              num_epochs=epoch, num_data_loader_workers=0)
        self.mdl.fit(training_dataset)

        cas = CoherenceUMASS(texts=[doc.split() for doc in preprocessed_documents],
                             topics=self.mdl.get_topic_lists(self.naspects))
        if len(cas.topics[0]) < 10:
            self.cas = 0
        else:
            self.cas = cas.score()

        # self.mdl.get_doc_topic_distribution(training_dataset, n_samples=20)

        # log_perplexity = -1 * np.mean(np.log(np.sum(bert, axis=0)))
        # self.perplexity = np.exp(log_perplexity)

        self.perplexity = 0
        pd.to_pickle(self.dict, f'{self.path}model.dict.pkl')
        pd.to_pickle(self.tp, f'{self.path}model.tp.pkl')
        self.mdl.save(f'{self.path[:-1]}/')
        self.mdl.model_dir = self.mdl._format_file()
        pd.to_pickle(self.mdl.model_dir, f'{self.path}model.path.pkl')
        # torch.save(self.mdl.model.state_dict(), f'{self.path}model.pth')

        with open(f'{self.path}model.perf.cas', 'wb') as f:
            pickle.dump(self.cas, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{self.path}model.perf.perplexity', 'wb') as f:
            pickle.dump(self.perplexity, f, protocol=pickle.HIGHEST_PROTOCOL)

    def preprocess(self, doctype, reviews):
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed(params.seed)
        np.random.seed(params.seed)
        random.seed(params.seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True

        reviews_ = [s for r in reviews for s in r.sentences]
        docs = [' '.join(text) for text in reviews_]
        stopwords = list(stop_words.words("english"))
        sp = WhiteSpacePreprocessingStopwords(docs, stopwords_list=stopwords)
        preprocessed_documents, unpreprocessed_corpus, vocab, retained_indices = sp.preprocess()
        return preprocessed_documents, unpreprocessed_corpus, vocab, retained_indices

    def show_topic(self, topic_id, nwords):
        return self.mdl.get_word_distribution_by_topic_id(topic_id)[0:nwords]

    def infer(self, doctype, review):
        # doc = [' '.join(text) for text in review]
        preprocessed_documents, unpreprocessed_corpus, vocab, retained_indices = self.preprocess(doctype, review)
        testing_dataset = self.tp.transform(text_for_contextual=unpreprocessed_corpus,
                                            text_for_bow=preprocessed_documents)
        review_aspects = self.mdl.get_doc_topic_distribution(testing_dataset, n_samples=10)
        output_list = []
        for lst in review_aspects:
            output_list.append([(i, v) for i, v in enumerate(lst)])
        return output_list
