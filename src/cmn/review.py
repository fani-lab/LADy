import pandas as pd
import copy

class Review(object):
    def __init__(self, id, sentences, time, author, aos, lempos):
        self.id = id
        self.sentences = sentences #list of sentences of list of tokens
        self.time = time
        self.author = author
        self.aos = aos #list of list of aspect_opinion_sentiment triples for per sentence, e.g., [[([7,8], [10, 11, 12], -1), ([15,17], [20], +1)]]
        self.lempos = lempos

    @staticmethod
    def load(path, output, settings): pass

    @staticmethod
    def to_df(reviews): return pd.DataFrame.from_dict([r.to_dict() for r in reviews])

    def to_dict(self): return {'id': self.id, 'sentences': self.sentences, 'aos': self.get_aos()}

    def get_aos(self):
        r = []
        for i, aos in enumerate(self.aos): r.append([([self.sentences[i][j] for j in a], [self.sentences[i][j] for j in o], s) for (a, o, s) in aos])
        return r

    def hide_aspects(self):
        r = copy.deepcopy(self)
        for i, sent in enumerate(r.sentences):
            # [sent.pop(k) for j, _, _ in r.aos[i] for k in j]
            for j, _, _ in r.aos[i]:
                for k in j: sent[k] = '#####'
        return r

    def preprocess(self): return self # note that any removal of words breakes the aos indexing!