import re, numpy as np, pandas as pd, random
import gensim

class AbstractAspectModel:
    stop_words = None
    def __init__(self, naspects, nwords):
        self.naspects = naspects
        self.nwords = nwords
        self.dict = None
        self.mdl = None
        self.cas = 0.00
        self.perplexity = 0.00

    def name(self): return self.__class__.__name__.lower()
    def load(self, path, settings): pass
    def train(self, reviews_train, reviews_valid, settings, doctype, no_extremes, output):
        corpus, self.dict = AbstractAspectModel.preprocess(doctype, reviews_train, no_extremes)
        self.dict.save(f'{output}model.dict')
        pd.to_pickle(self.cas, f'{output}model.perf.cas')
        pd.to_pickle(self.perplexity, f'{output}model.perf.perplexity')

    def quality(self, metric):
        result = {"coherence": f'{np.mean(self.cas)}\u00B1{np.std(self.cas)}', "perplexity": self.perplexity}
        return result[metric]
        # elif metric is "perplexity":
        #     return

    def get_aspects_words(self, nwords): pass
    def get_aspect_words(self, aspect_id, nwords): pass
    def infer(self, review, doctype): pass
    def infer_batch(self, reviews_test, h_ratio, doctype, settings):
        pairs = []
        for r in reviews_test:
            r_aspects = [[w for a, o, s in sent for w in a] for sent in r.get_aos()]  # [['service', 'food'], ['service'], ...]
            if len(r_aspects[0]) == 0: continue  # ??
            if random.random() < h_ratio: r_ = r.hide_aspects()
            else: r_ = r
            r_pred_aspects = self.infer(r_, doctype)
            # removing duplicate aspect words ==> handled in metrics()
            pairs.extend(list(zip(r_aspects, self.merge_aspects_words(r_pred_aspects, self.nwords))))
        return pairs

    def merge_aspects_words(self, r_pred_aspects, nwords):
        # Since predicted aspects are distributions over words, we need to flatten them into list of words.
        # Given a and b are two aspects, we do prob(a) * prob(a_w) for all w \in a and prob(b) * prob(b_w) for all w \in b
        # Then sort.
        result = []
        for subr_pred_aspects in r_pred_aspects:
            subr_pred_aspects_words = [w_p
                                   for l in [[(w, a_p * w_p) for w, w_p in self.get_aspect_words(a, nwords)]
                                             for a, a_p in subr_pred_aspects]
                                   for w_p in l]
            result.append(sorted(subr_pred_aspects_words, reverse=True, key=lambda t: t[1]))
        return result

    @staticmethod
    def preprocess(doctype, reviews, settings=None):
        if not AbstractAspectModel.stop_words:
            import nltk
            AbstractAspectModel.stop_words = nltk.corpus.stopwords.words('english')

        reviews_ = []
        if doctype == 'rvw': reviews_ = [np.concatenate(r.sentences) for r in reviews]
        elif doctype == 'snt': reviews_ = [s for r in reviews for s in r.sentences]
        reviews_ = [[word for word in doc if word not in AbstractAspectModel.stop_words and len(word) > 3 and re.match('[a-zA-Z]+', word)] for doc in reviews_]
        dict = gensim.corpora.Dictionary(reviews_)
        if settings: dict.filter_extremes(no_below=settings['no_below'], no_above=settings['no_above'], keep_n=100000)
        dict.compactify()
        return reviews_, dict

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