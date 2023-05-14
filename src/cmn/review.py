import pandas as pd, copy
from scipy.spatial.distance import cosine

class Review(object):
    translator_mdl = None; translator_tokenizer = None
    semantic_mdl = None
    def __init__(self, id, sentences, time=None, author=None, aos=None, lempos=None, parent=None, lang='eng_Latn', category=None):
        self.id = id
        self.sentences = sentences #list of sentences of list of tokens
        self.time = time
        self.author = author
        self.aos = aos #list of list of aspect_opinion_sentiment triples for per sentence, e.g., [[([7,8], [10, 11, 12], -1), ([15,17], [20], +1)]]
        self.lempos = lempos
        self.lang = lang
        self.category = category

        self.parent = parent
        self.augs = {} #distionary of translated and backtranslated augmentations of this review in object format, e.g.,
        # {'deu_Latn': (Review1(self.id, 'dies ist eine bewertung', None, None, None, None, self, 'deu_Latn'),
        #               Review2(self.id, 'this is a review', None, None, None, None, self, 'eng_Latn'))

    def to_dict(self):
        result = [{'id': self.id, 'sentences': self.sentences, 'aos': self.get_aos(), 'lang': self.lang, 'orig': False if self.parent else True}]
        for k in self.augs:
            #result.append(self.augs[k][0].to_dict())
            result.append(self.augs[k][1].to_dict())
        return result

    def get_aos(self):
        r = []
        for i, aos in enumerate(self.aos): r.append([([self.sentences[i][j] for j in a], [self.sentences[i][j] for j in o], s) for (a, o, s) in aos])
        return r

    def get_txt(self): return '. '.join(' '.join(s) for s in self.sentences)

    def get_sentences(self): return self.sentences

    def get_category(self): return self.category
    
    def hide_aspects(self):
        r = copy.deepcopy(self)
        for i, sent in enumerate(r.sentences):
            # [sent.pop(k) for j, _, _ in r.aos[i] for k in j]
            for j, _, _ in r.aos[i]:
                for k in j: sent[k] = '#####'
        return r

    def preprocess(self): return self # note that any removal of words breakes the aos indexing!

    def translate(self, tgt, settings):
        src = self.lang
        if not Review.translator_mdl:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            Review.translator_mdl = AutoModelForSeq2SeqLM.from_pretrained(settings['nllb'])
            Review.translator_tokenizer = AutoTokenizer.from_pretrained(settings['nllb'])

        from transformers import pipeline
        Review.translator = pipeline("translation", model=Review.translator_mdl, tokenizer=Review.translator_tokenizer, src_lang=src, tgt_lang=tgt, max_length=settings['max_l'], device=settings['device'])
        Review.back_translator = pipeline("translation", model=Review.translator_mdl, tokenizer=Review.translator_tokenizer, src_lang=tgt, tgt_lang=src, max_length=settings['max_l'], device=settings['device'])

        translated_txt = Review.translator(self.get_txt())[0]['translation_text']
        translated_obj = Review(self.id, [[str(t).lower() for t in translated_txt.split()]], None, None, None, None, self, tgt)

        back_translated_txt = Review.back_translator(translated_txt)[0]['translation_text']
        back_translated_obj = Review(self.id, [[str(t).lower() for t in back_translated_txt.split()]], None, None, None, None, self, src)

        self.augs[tgt] = (translated_obj, back_translated_obj, self.semsim(back_translated_obj))
        return self.augs[tgt]

    def semsim(self, other):
        if not Review.semantic_mdl:
            from sentence_transformers import SentenceTransformer
            Review.semantic_mdl = SentenceTransformer("johngiorgi/declutr-small")
        me, you = Review.semantic_mdl.encode([self.get_txt(), other.get_txt()])
        return 1 - cosine(me, you)

    @staticmethod
    def load(path): pass

    @staticmethod
    def to_df(reviews): return pd.DataFrame.from_dict([rr for r in reviews for rr in r.to_dict()])

    @staticmethod
    def translate_batch(reviews, tgt, settings):
        src = reviews[0].lang
        if not Review.translator_mdl:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            Review.translator_mdl = AutoModelForSeq2SeqLM.from_pretrained(settings['nllb'])
            Review.translator_tokenizer = AutoTokenizer.from_pretrained(settings['nllb'])

        from transformers import pipeline
        print("pre-translator models")
        translator = pipeline("translation", model=Review.translator_mdl, tokenizer=Review.translator_tokenizer, src_lang=src, tgt_lang=tgt, max_length=settings['max_l'], device=settings['device'])
        back_translator = pipeline("translation", model=Review.translator_mdl, tokenizer=Review.translator_tokenizer, src_lang=tgt, tgt_lang=src, max_length=settings['max_l'], device=settings['device'])

        print("review")
        reviews_txt = [r.get_txt() for r in reviews]
        print("translating")
        translated_txt = translator(reviews_txt)
        print("traslating complete")
        back_translated_txt = back_translator([r_['translation_text'] for r_ in translated_txt])

        for i, r in enumerate(reviews):
            print("obj")
            translated_obj = Review(r.id, [[str(t).lower() for t in translated_txt[i]['translation_text'].split()]], None, None, None, None, r, tgt)
            back_translated_obj = Review(r.id, [[str(t).lower() for t in back_translated_txt[i]['translation_text'].split()]], None, None, None, None, r, src)
            r.augs[tgt] = (translated_obj, back_translated_obj, r.semsim(back_translated_obj))


