review = 'this is a sample'

model_name = 'rnd' #could be ['rnd', 'lda', 'btm', 'ctm', 'octis.ctm', 'octis.neurallda']
naspects = 5 #could be [5, 10, 15, 20, 25]
backtrans_lang = '' #could be ['', 'pes_Arab', 'zho_Hans', 'deu_Latn', 'arb_Arab', 'fra_Latn', 'spa_Latn']
domain = 'restaurant' #could be [restaurant, laptop]
nwords = 20 #fixed

for model_name in ['rnd', 'lda', 'btm', 'ctm', 'octis.ctm', 'octis.neurallda']:
    for naspects in [5, 10, 15, 20, 25]:
        for backtrans_lang in ['', 'pes_Arab', 'zho_Hans', 'deu_Latn', 'arb_Arab', 'fra_Latn', 'spa_Latn']:
            for domain in ['restaurant']:#, 'laptop']:
                model_path = '../output/semeval+/SemEval-14/Semeval-14-Restaurants_Train.xml' if domain == 'restaurant' else '../output/semeval+/SemEval-14/Laptop_Train_v2.xml'

                if "rnd" == model_name: from aml.rnd import Rnd; am = Rnd(naspects, nwords)
                if "lda" == model_name: from aml.lda import Lda; am = Lda(naspects, nwords)
                if "btm" == model_name: from aml.btm import Btm; am = Btm(naspects, nwords)
                if "ctm" == model_name: from aml.ctm import Ctm; am = Ctm(naspects, nwords, 768, 10)
                if "octis.ctm" == model_name: from octis.models.CTM import CTM; from aml.nrl import Nrl; am = Nrl(CTM(), naspects,nwords,None)
                if "octis.neurallda" == model_name: from octis.models.NeuralLDA import NeuralLDA; from aml.nrl import Nrl; am = Nrl(NeuralLDA(), naspects, nwords, None)

                from cmn.review import Review
                r = Review(id=0, sentences=[review.split()], time=None, author=None, aos=[[([0],[],0)]], lempos=None, parent=None, lang='eng_Latn', category=None)
                am.load(f'{model_path}/{naspects}{"." + backtrans_lang if backtrans_lang else ""}/{am.name()}/f0.')
                print(am.infer_batch(reviews_test=[r], h_ratio=0.0, doctype='snt')[0][1][:naspects])

