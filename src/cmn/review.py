import pandas as pd, copy, numpy as np
from scipy.spatial.distance import cosine

class Review(object):
    translator_mdl = None; translator_tokenizer = None
    semantic_mdl = None; align_mdl = None
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
        #               Review2(self.id, 'this is a review', None, None, None, None, self, 'eng_Latn'),
        #               semantic_similarity_score)

    def to_dict(self, w_augs=False):
        result = [{'id': self.id,
                   'text': self.get_txt(),
                   'sentences': self.sentences,
                   'aos': self.get_aos(), #self.parent.get_aos() if self.parent else self.get_aos(),
                   'lang': self.lang,
                   'orig': False if self.parent else True}]
        if not w_augs: return result
        for k in self.augs:
            #result += self.augs[k][0].to_dict()
            result += self.augs[k][1].to_dict()
        return result

    def get_aos(self):
        r = []
        if not self.aos: return r
        for i, aos in enumerate(self.aos): r.append([([self.sentences[i][j] for j in a], [self.sentences[i][j] for j in o], s) for (a, o, s) in aos])
        return r

    def get_txt(self): return '. '.join(' '.join(s) for s in self.sentences)

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
        translated_obj = Review(id=self.id, sentences=[[str(t).lower() for t in translated_txt.split()]], parent=self, lang=tgt, time=None, author=None, aos=None)
        translated_obj.aos, _ = self.semalign(translated_obj)

        back_translated_txt = Review.back_translator(translated_txt)[0]['translation_text']
        back_translated_obj = Review(id=self.id, sentences=[[str(t).lower() for t in back_translated_txt.split()]], parent=self, lang=src, time=None, author=None, aos=None)
        back_translated_obj.aos, _ = self.semalign(back_translated_obj)

        self.augs[tgt] = (translated_obj, back_translated_obj, self.semsim(back_translated_obj))
        return self.augs[tgt]

    def semsim(self, other):
        if not Review.semantic_mdl:
            from sentence_transformers import SentenceTransformer
            Review.semantic_mdl = SentenceTransformer("johngiorgi/declutr-small")
        me, you = Review.semantic_mdl.encode([self.get_txt(), other.get_txt()])
        return 1 - cosine(me, you)

    def semalign(self, other):
        if not Review.align_mdl:
            from simalign import SentenceAligner
            Review.align_mdl = SentenceAligner(model="bert", token_type="bpe", matching_methods="i")
        aligns = [Review.align_mdl.get_word_aligns(s1, o1)['itermax'] for s1, o1 in zip(self.sentences, other.sentences)]
        other_aos = []
        for i, (aos, _) in enumerate(zip(self.aos, self.sentences)):
            for (a, o, s) in aos:
                other_a = [idx2 for idx in a for idx1, idx2 in aligns[i] if idx == idx1]
                other_a.sort()
                other_aos.append((other_a, o, s))
        return other_aos, aligns

    def get_lang_stats(self):
        import nltk
        from rouge import Rouge
        from sklearn.metrics import accuracy_score

        result = {}
        r = self.get_txt()
        result['r_ntoken'] = len(r.split())
        for lang in self.augs.keys():
            r_ = self.augs[lang][1].get_txt()
            # r_ = r #for testing purpose => should be very close to 1 for all metrics
            result[lang + '_r_backtrans_ntoken'] = len(r_.split())
            result[lang + '_semsim'] = self.augs[lang][2]
            result[lang + '_bleu'] = np.mean(nltk.translate.bleu_score.sentence_bleu([r.split()], r_.split(), weights=[(1 / bleu_no,) * bleu_no for bleu_no in range(1, min(4, result['r_ntoken'] + 1))]))
            # https://pypi.org/project/rouge/
            result[lang + '_rouge_f'] = np.mean([v['f'] for k, v in Rouge(metrics=[f'rouge-{i+1}' for i in range(0, min(5, len(r.split())))]).get_scores(r_, r)[0].items()])
            # we need to make r_ as equal size as r
            result[lang + '_em'] = accuracy_score(r.split(), r_.split()[:result['r_ntoken']] if len(r_.split()) > result['r_ntoken'] else r_.split() + [''] * (result['r_ntoken'] - len(r_.split())))
        return result

    @staticmethod
    def load(path): pass

    @staticmethod
    def to_df(reviews, w_augs=False): return pd.DataFrame.from_dict([rr for r in reviews for rr in r.to_dict(w_augs)])

    @staticmethod
    def translate_batch(reviews, tgt, settings):
        src = reviews[0].lang
        if not Review.translator_mdl:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            Review.translator_mdl = AutoModelForSeq2SeqLM.from_pretrained(settings['nllb'])
            Review.translator_tokenizer = AutoTokenizer.from_pretrained(settings['nllb'])

        from transformers import pipeline
        translator = pipeline("translation", model=Review.translator_mdl, tokenizer=Review.translator_tokenizer, src_lang=src, tgt_lang=tgt, max_length=settings['max_l'], device=settings['device'])
        back_translator = pipeline("translation", model=Review.translator_mdl, tokenizer=Review.translator_tokenizer, src_lang=tgt, tgt_lang=src, max_length=settings['max_l'], device=settings['device'])

        reviews_txt = [r.get_txt() for r in reviews]
        translated_txt = translator(reviews_txt)
        back_translated_txt = back_translator([r_['translation_text'] for r_ in translated_txt])

        for i, r in enumerate(reviews):
            translated_obj = Review(id=r.id, sentences=[[str(t).lower() for t in translated_txt[i]['translation_text'].split()]], parent=r, lang=tgt, time=None, author=None, aos=None, lempos=None,)
            translated_obj.aos, _ = r.semalign(translated_obj)

            back_translated_obj = Review(id=r.id, sentences=[[str(t).lower() for t in back_translated_txt[i]['translation_text'].split()]], parent=r, lang=src, time=None, author=None, aos=r.aos, lempos=None,)
            r.augs[tgt] = (translated_obj, back_translated_obj, r.semsim(back_translated_obj))

    @staticmethod
    def get_stats(datapath, output, cache=True, plot=True, plot_title=None):
        try:
            print(f'Loading the stats pickle from {datapath}...')
            if not cache: raise FileNotFoundError
            stats = pd.read_pickle(f'{output}/stats.pkl')
            if plot: Review.plot_dist(stats, output, plot_title)
        except FileNotFoundError:
            print(f'File {datapath} not found! Generating stats ...')
            reviews = pd.read_pickle(datapath)
            from collections import Counter
            stats = {'*nreviews': len(reviews), '*naspects': 0, '*ntokens': 0}
            asp_nreviews = Counter()        # aspects : number of reviews that contains the aspect
            token_nreviews = Counter()      # tokens : number of reviews that contains the token
            nreviews_naspects = Counter()   # v number of reviews with 1 aspect, ..., k aspects, ...
            nreviews_ntokens = Counter()    # v number of reviews with 1 token, ...,  k tokens, ...
            nreviews_category = Counter()  # v number of categories with 1 review, ..., k reviews, ...
            reviews_lang_stats = []

            for r in reviews:
                r_aspects = r.get_aos()[0]
                r_tokens = [token for sentence in r.sentences for token in sentence]
                asp_nreviews.update(' '.join(a) for (a, o, s) in r_aspects)
                token_nreviews.update(token for token in r_tokens)
                nreviews_naspects.update([len(r_aspects)])
                nreviews_ntokens.update([len(r_tokens)])
                if hasattr(r, 'category'): nreviews_category.update(r.category)

                reviews_lang_stats.append(r.get_lang_stats())

            naspects_nreviews = Counter(asp_nreviews.values())   # v number of aspects with 1 review, ..., k reviews, ...
            ntokens_nreviews = Counter(token_nreviews.values())  # v number of tokens with 1 review, ..., k reviews, ...
            stats["*naspects"] = len(asp_nreviews.keys()) # unique. Non-unique number of aspects: sum(asp_nreviews.values())
            stats["*ntokens"] = len(token_nreviews.keys()) # unique. Non-unique number of tokens: sum(token_nreviews.values())
            stats['nreviews_naspects'] = {k: v for k, v in sorted(nreviews_naspects.items(), key=lambda item: item[1], reverse=True)}
            stats['nreviews_ntokens'] = {k: v for k, v in sorted(nreviews_ntokens.items(), key=lambda item: item[1], reverse=True)}
            stats['naspects_nreviews'] = {k: v for k, v in sorted(naspects_nreviews.items(), key=lambda item: item[1], reverse=True)}
            stats['ntokens_nreviews'] = {k: v for k, v in sorted(ntokens_nreviews.items(), key=lambda item: item[1], reverse=True)}
            stats['nreviews_category'] = {k: v for k, v in sorted(nreviews_category.items(), key=lambda item: item[1], reverse=True)}
            stats['*avg_ntokens_review'] = sum(k * v for k, v in nreviews_ntokens.items()) / sum(nreviews_ntokens.values()) # average number of tokens per review
            stats['*avg_naspects_review'] = sum(k * v for k, v in nreviews_naspects.items()) / sum(nreviews_naspects.values()) # average number of aspects per review
            stats['*avg_lang_stats'] = pd.DataFrame.from_dict(reviews_lang_stats).mean().to_dict()
            if output: pd.to_pickle(stats, f'{output}/stats.pkl')
            if plot: Review.plot_dist(stats, output, plot_title)
        import json
        print(json.dumps(stats, indent=4))
        # print(stats)
        return stats

    @staticmethod
    def plot_dist(stats, output, plot_title):
        from matplotlib import pyplot as plt
        plt.rcParams.update({'font.family': 'Consolas'})
        print("plotting distribution data ...")
        for k, v in stats.items():
            if (not k.startswith("*")): # the * values cannot be plotted
                fig = plt.figure(k, figsize=(1.5, 1.5))
                ax = fig.add_subplot(1, 1, 1)
                ax.set_facecolor('whitesmoke')
                ax.loglog(*zip(*stats[k].items()), marker='x', linestyle='None', markeredgecolor='m')
                ax.set_xlabel(k.split('_')[1][0].replace('n', '#') + k.split('_')[1][1:])
                ax.set_ylabel(k.split('_')[0][0].replace('n', '#') + k.split('_')[0][1:])
                ax.grid(True, color="#93a1a1", alpha=0.3)
                ax.minorticks_off()
                ax.xaxis.set_tick_params(size=2, direction='in')
                ax.yaxis.set_tick_params(size=2, direction='in')
                
                # wrapping labels
                labels = []
                for l in ax.get_xticklabels(): 
                    l.set_text('\n#'.join(l.get_text().split("#")))
                    labels.append(l)
                ax.set_xticklabels(labels, ha='left')

                ax.xaxis.get_label().set_size(12)
                ax.yaxis.get_label().set_size(12)
                ax.set_title(plot_title)
                fig.savefig(f'{output}/{k}.pdf', dpi=100, bbox_inches='tight')
                plt.show()

    @staticmethod
    def plot_semsim_dist(datapath, output, plot_title):
        from matplotlib import pyplot as plt
        plt.rcParams.update({'font.family': 'Consolas'})
        import seaborn as sns
        reviews = pd.read_pickle(datapath)

        hist_dict = [{'original': 'eng_Latn', 'target': Review.lang_title(k), 'score': v[2]} for r in reviews for k, v in r.augs.items()]
        df = pd.DataFrame.from_dict(hist_dict)
        fig = plt.figure(figsize=(6, 2))
        ax = fig.add_subplot(1, 1, 1)
        x_range = [i / 10 for i in range(0, 11)]
        # ax.set_ylim([0, len(reviews)])
        # plt.yscale('log')
        plt.ylabel('#reviews')
        plt.xlabel('similarity score')
        ax.set_title(plot_title, x=0.2, y=0.8, fontsize=11)
        ax.set_facecolor('whitesmoke')
        h = sns.histplot(df,
                         x='score',
                         hue='target',
                         element='step', #also try 'poly'
                         stat='density',
                         common_norm=False)
        sns.move_legend(ax, 'upper left')

        plt.legend([])

        h.legend_.set_title(None)
        h.set(xticks=x_range)
        plt.savefig(f'{output}', dpi=100, bbox_inches='tight')
        # df.to_csv(f'{output.replace("pdf", "csv")}')
        plt.show()
        # plt.clf()

    @staticmethod
    def lang_title(lang_code):
        if   lang_code == 'zho_Hans': return 'chinese'
        elif lang_code == 'deu_Latn': return 'german'
        elif lang_code == 'fra_Latn': return 'french'
        elif lang_code == 'arb_Arab': return 'arabic'
        elif lang_code == 'pes_Arab': return 'farsi'
        elif lang_code == 'spa_Latn': return 'spanish'
        elif lang_code == 'eng_Latn': return 'english'
        elif lang_code == 'pes_Arab.zho_Hans.deu_Latn.arb_Arab.fra_Latn.spa_Latn': return 'all'
        elif lang_code == None: return ['pes_Arab', 'zho_Hans', 'deu_Latn', 'arb_Arab', 'fra_Latn', 'spa_Latn', 'all']

