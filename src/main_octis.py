import argparse, os, multiprocessing, time, string
import numpy as np, random, pandas as pd

from octis.models.model import *
from octis.preprocessing.preprocessing import Preprocessing
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

import params
from cmn.review import Review
import main as lady

def create_octis_ds(splits, reviews, f, output):
    if not os.path.isdir(output): os.makedirs(output)

    reviews_train = np.array(reviews)[splits['folds'][f]['train']].tolist()
    reviews_train.extend([r_.augs[lang][1] for r_ in reviews_train for lang in params.settings['prep']['langaug'] if lang and r_.augs[lang][2] >= params.settings['train']['langaug_semsim']])

    df_train = Review.to_df(reviews_train, w_augs=False)
    df_train['part'] = 'train'

    reviews_valid = np.array(reviews)[splits['folds'][f]['valid']].tolist()
    df_valid = Review.to_df(reviews_valid)
    df_valid['part'] = 'val'

    reviews_test = np.array(reviews)[splits['test']].tolist()
    df_test = Review.to_df(reviews_test)
    df_test['part'] = 'test'

    df = pd.concat([df_train, df_valid, df_test])
    df.to_csv(f'{output}/corpus.tsv', sep='\t', encoding='utf-8', index=False, columns=['text', 'part'], header=None)

def main(args):
    if not os.path.isdir(args.output): os.makedirs(args.output)
    langaug_str = '.'.join([l for l in params.settings['prep']['langaug'] if l])
    reviews = lady.load(args.data, f'{args.output}/reviews.{langaug_str}.pkl'.replace('..pkl', '.pkl'))
    splits = lady.split(len(reviews), args.output)
    output = f'{args.output}/{args.naspects}.{langaug_str}'.rstrip('.')

    if 'train' in params.settings['cmd']:
        from octis.dataset.dataset import Dataset
        # if "rnd" == args.am: from aml.rnd import Rnd; am = Rnd(args.naspects)
        # if "lda" == args.am: from octis.models.LDA import LDA; am = LDA() ==> octis uses the single thread LdaModel instead of LdaMulticore!
        # if "btm" == args.am: from aml.btm import Btm; am = Btm(args.naspects)
        if "ctm" == args.am: from octis.models.CTM import CTM; am = CTM(inference_type='combined');
        if "nrl" == args.am: from octis.models.NeuralLDA import NeuralLDA; am = NeuralLDA()

        am.hyperparameters.update(params.settings['train'][args.am])
        am.hyperparameters.update({'num_topics': args.naspects})

        print(f'\n2. Aspect model training for {args.am} ...')
        print('#' * 50)

        for f in splits['folds'].keys():
            octis_ds = f'{output}/octis/f{f}/'
            octis_model_output = f'{output}/octis/{args.am}/'
            if 'bert_path' in am.hyperparameters.keys(): am.hyperparameters['bert_path'] = octis_ds
            try:
                print(f'2.1. Loading saved aspect model from {octis_model_output}f{f}.model.npz ...')
                am = load_model_output(f'{octis_model_output}f{f}.model.npz', top_words=params.settings['train']['nwords'])
                am['metrics'] = pd.read_pickle(f'{octis_model_output}f{f}.model.perf.cas')
            except (FileNotFoundError, EOFError) as e:
                print(f'2.1. Loading saved aspect model failed! Training {args.am} for {args.naspects} of aspects ...')
                dataset = Dataset()
                try: dataset.load_custom_dataset_from_folder(octis_ds)
                except:
                    create_octis_ds(splits, reviews, f, octis_ds)
                    dataset.load_custom_dataset_from_folder(octis_ds)

                # preprocessor = Preprocessing(vocabulary=None, max_features=None, remove_punctuation=True, punctuation=string.punctuation, lemmatize=True, stopword_list='english', min_chars=1, min_words_docs=0)
                # dataset = preprocessor.preprocess_dataset(documents_path=r'./corpus.tsv')
                # dataset.save('hello_dataset')
                # training, valid, test

                # training
                t_s = time.time()
                am.use_partitions = True
                am.update_with_test = True
                am_output = am.train_model(dataset, top_words=params.settings['train']['nwords'])
                print(f'Trained time elapsed including language augs {params.settings["prep"]["langaug"]}: {time.time() - t_s}')

                if not os.path.isdir(octis_model_output): os.makedirs(octis_model_output)
                save_model_output(am_output, f'{octis_model_output}f{f}.model')
                metrics = {}
                for m in ['u_mass', 'c_v', 'c_uci', 'c_npmi']: metrics[m] = Coherence(topk=params.settings['train']['nwords'], measure=m, processes=params.settings['train'][args.am]['ncore']).score(am_output)
                pd.to_pickle(metrics, f'{octis_model_output}f{f}.model.perf.cas')

            # testing

    # evaluating
    if 'eval' in params.settings['cmd']:
        df_f_means = pd.DataFrame()
        for f in splits['folds'].keys():
            input = f'{output}f{f}.model.pred.{params.settings["test"]["h_ratio"]}'
            df_mean = lady.evaluate(input, f'{input}.eval.mean.csv')
            df_f_means = pd.concat([df_f_means, df_mean], axis=1)
        df_f_means.mean(axis=1).to_frame('mean').to_csv(f'{output}model.pred.eval.mean.{params.settings["test"]["h_ratio"]}.csv')

# {CUDA_VISIBLE_DEVICES=0,1} won't work https://discuss.pytorch.org/t/using-torch-data-prallel-invalid-device-string/166233
# TOKENIZERS_PARALLELISM=true
# TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0 python -u main_octis.py -am lda -naspect 5 -data ../data/raw/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml -output ../output/semeval+/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml 2>&1 | tee ../output/semeval+/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml/log.txt &
# TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0 python -u main_octis.py -am lda -naspect 5 -data ../data/raw/semeval/SemEval-14/Semeval-14-Restaurants_Train.xml -output ../output/semeval+/SemEval-14/Semeval-14-Restaurants_Train.xml 2>&1 | tee ../output/semeval+/SemEval-14/Semeval-14-Restaurants_Train.xml/log.txt &
# TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0 python -u main_octis.py -am lda -naspect 5 -data ../data/raw/semeval/2015SB12/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml -output ../output/semeval+/2015SB12/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml 2>&1 | tee ../output/semeval+/2015SB12/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml/log.txt &
# TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0 python -u main_octis.py -am lda -naspect 5 -data ../data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml -output ../output/semeval+/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml 2>&1 | tee ../output/semeval+/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml/log.txt &

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latent Aspect Detection')
    parser.add_argument('-am', type=str.lower, default='rnd', help='aspect modeling method (eg. --am lda)')
    parser.add_argument('-data', dest='data', type=str, help='raw dataset file path, e.g., -data ..data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml')
    parser.add_argument('-output', dest='output', type=str, default='../output/', help='output path, e.g., -output ../output/semeval/2016.xml')
    parser.add_argument('-naspects', dest='naspects', type=int, default=25, help='user-defined number of aspects, e.g., -naspect 25')
    args = parser.parse_args()

    main(args)
    if 'agg' in params.settings['cmd']: main.agg(args.output, args.output)
