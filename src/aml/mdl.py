import re
import numpy as np


class AbstractAspectModel:
    stop_words = None
    def __init__(self, naspects): self.naspects = naspects

    def load(self): pass
    def train(self, reviews_train, reviews_valid, settings, doctype, output): pass
    def get_aspects(self, nwords): pass
    def get_aspect(self, nwords): pass
    def infer(self, doctype, review): pass
    def get_aspects_words(self, r_pred_aspects, nwords):
        result = []
        for subr_pred_aspects in r_pred_aspects:
            subr_pred_aspects_words = [w_p
                                   for l in [[(w, a_p * w_p) for w, w_p in self.get_aspect(a, nwords)]
                                             for a, a_p in subr_pred_aspects]
                                   for w_p in l]
            result.append(sorted(subr_pred_aspects_words, reverse=True, key=lambda t: t[1]))
        return result

    @staticmethod
    def preprocess(doctype, reviews):
        if not AbstractAspectModel.stop_words:
            import nltk
            AbstractAspectModel.stop_words = nltk.corpus.stopwords.words('english')

        reviews_ = []
        if doctype == 'rvw': reviews_ = [np.concatenate(r.sentences) for r in reviews]
        elif doctype == 'snt': reviews_ = [s for r in reviews for s in r.sentences]
        return [[word for word in doc if word not in AbstractAspectModel.stop_words and len(word) > 3 and re.match('[a-zA-Z]+', word)] for doc in reviews_]

    @staticmethod
    def quality(model, metric):
        result = {"Coherence": f'{np.mean(model.cas)}\u00B1{np.std(model.cas)}',
                  "Perplexity": model.perplexity}
        return result[metric]
        # elif metric is "perplexity":
        #     return

    # @staticmethod
    # def plot_coherence(path, cas):
    # dict of coherences for different naspects, e.g., {'2': [0.3, 0.5], '3': [0.3, 0.5, 0.7]}.
    #     # np.mean(row wise)
    #     # np.std(row wise)
    #
    #     # plt.plot(x, mean, '-or', label='mean')
    #     # plt.xlim(start - 0.025, limit - 1 + 0.025)
    #     plt.xlabel("#aspects")
    #     plt.ylabel("coherence")
    #     plt.legend(loc='best')
    #     plt.savefig(f'{path}coherence.png')
    #     plt.clf()