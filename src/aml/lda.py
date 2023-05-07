import gensim, logging, pickle
from gensim.models.callbacks import PerplexityMetric, ConvergenceMetric, CoherenceMetric
from gensim.models.coherencemodel import CoherenceModel

from .mdl import AbstractAspectModel

# @inproceedings{DBLP:conf/naacl/BrodyE10,
#   author       = {Samuel Brody and Noemie Elhadad},
#   title        = {An Unsupervised Aspect-Sentiment Model for Online Reviews},
#   booktitle    = {Human Language Technologies: Conference of the North American Chapter of the Association of Computational Linguistics, Proceedings, June 2-4, 2010, Los Angeles, California, {USA}},
#   pages        = {804--812},
#   publisher    = {The Association for Computational Linguistics},
#   year         = {2010},
#   url          = {https://aclanthology.org/N10-1122/},
#   biburl       = {https://dblp.org/rec/conf/naacl/BrodyE10.bib},
# }
class Lda(AbstractAspectModel):
    def __init__(self, naspects): super().__init__(naspects)

    def load(self, path):
        self.mdl = gensim.models.LdaModel.load(f'{path}model')
        assert self.mdl.num_topics == self.naspects
        self.dict = gensim.corpora.Dictionary.load(f'{path}model.dict')
        with open(f'{path}model.perf.cas', 'rb') as f: self.cas = pickle.load(f)
        with open(f'{path}model.perf.perplexity', 'rb') as f: self.perplexity = pickle.load(f)

    def train(self, reviews_train, reviews_valid, settings, doctype, langaug, output):
        reviews_ = super().preprocess(doctype, reviews_train)
        self.dict = gensim.corpora.Dictionary(reviews_)
        if settings['no_extremes']: self.dict.filter_extremes(no_below=settings['no_extremes']['no_below'], no_above=settings['no_extremes']['no_above'], keep_n=100000)
        self.dict.compactify()
        corpus = [self.dict.doc2bow(doc) for doc in reviews_]

        logging.getLogger().handlers.clear()
        logging.basicConfig(filename=f'{output}model.train.log', format="%(asctime)s:%(levelname)s:%(message)s", level=logging.NOTSET)
        # callback functions cannot be applied in parallel lda, only for LdaModel()
        # perplexity_logger = PerplexityMetric(corpus=corpus, logger='shell')
        # convergence_logger = ConvergenceMetric(logger='shell')
        # coherence_cv_logger = CoherenceMetric(corpus=corpus, logger='shell', coherence='c_v', texts=reviews_)
        # self.model = gensim.models.wrappers.LdaMallet(mallet, corpus, num_topics=self.naspects, id2word=self.dict, workers=cores, iterations=iter, callback=)
        # alpha=symetric, i.e., 1/#topics, beta=0.01
        self.mdl = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics=self.naspects, id2word=self.dict, workers=settings['ncore'], passes=settings['passes'], random_state=settings['seed'], per_word_topics=True)

        # TODO: quality diagram ==> https://www.meganstodel.com/posts/callbacks/
        aspects, probs = self.get_aspects(settings['nwords'])
        # https://stackoverflow.com/questions/50607378/negative-values-evaluate-gensim-lda-with-topic-coherence
        # umass: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://aclanthology.org/D11-1024.pdf
        # [-inf, 0]: close to zero, the better
        self.cas = CoherenceModel(model=self.mdl, topics=aspects, corpus=corpus, dictionary=self.dict, coherence='u_mass', texts=reviews_).get_coherence_per_topic()
        self.perplexity = self.mdl.log_perplexity(corpus)
        self.dict.save(f'{output}model.dict')
        self.mdl.save(f'{output}model')
        with open(f'{output}model.perf.cas', 'wb') as f: pickle.dump(self.cas, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{output}model.perf.perplexity', 'wb') as f: pickle.dump(self.perplexity, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_aspects_words(self, nwords):
        # self.model.get_topics() does not have words
        # self.model.show_topics() and model.show_topic()
        words = []
        probs = []
        for idx, aspect in self.mdl.print_topics(-1, num_words=nwords):
            words.append([]); probs.append([])
            words_probs = aspect.split('+')
            for word_prob in words_probs:
                if any(char.isdigit() for char in word_prob):
                    probs[-1].append(word_prob.split('*')[0])
                    words[-1].append(word_prob.split('*')[1].split('"')[1])
                else:
                    probs[-1].append(0.0)
                    words[-1].append(word_prob.replace('"', ''))
        return words, probs

    def get_aspect_words(self, aspect_id, nwords): return self.mdl.show_topic(aspect_id, nwords)

    def infer(self, review, doctype):
        review_aspects = []
        review_ = super().preprocess(doctype, [review])
        for r in review_: review_aspects.append(self.mdl.get_document_topics(self.dict.doc2bow(r), minimum_probability=self.mdl.minimum_probability))
        return review_aspects



