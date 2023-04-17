import argparse, os, pickle
from time import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import sys
import pandas as pd
from tqdm import tqdm
import nltk
from rouge import Rouge

def tokenize(raw_corpus, flag):
    corpus = []
    for sentence in raw_corpus:
        # each sentence is one reference/hypothesis
        corpus.append([sentence.split()] if (flag == 'r') else sentence.split())
    return corpus


def load(ref_path, can_path):
    ref_raw = pd.read_csv(f'{ref_path}')['sentences'].tolist()
    can_raw = pd.read_csv(f'{can_path}')['sentences'].tolist()
    assert len(ref_raw) == len(can_raw), "Datasets should have the same length"

    ref_corpus = tokenize(ref_raw, 'r')
    can_corpus = tokenize(can_raw, 'c')
    assert len(ref_corpus) == len(can_corpus), "Tokenized corpus should have the same length"

    return ref_corpus, can_corpus

def calculate_bleu(ref_path, can_path):
    ref_corpus, can_corpus = load(ref_path, can_path)
    score = nltk.translate.bleu_score.corpus_bleu(ref_corpus, can_corpus) # default 4-gram
    print(score)

def main(args):
    #path = f'{args.output}/augmentation/back-translation'
    #if not os.path.isdir(path): os.makedirs(path)
    for l in args.lan:
        og_data = '../../data/augmentation/back-translation/D.csv' # Original Dataset
        trans_data = f'../../data/augmentation/back-translation/D.{l}.csv' # Translated Dataset
        back_trans_data = f'../../data/augmentation/back-translation/D_{l}.csv' # Back-translated Dataset

        calculate_bleu(og_data, trans_data)
        calculate_bleu(trans_data, back_trans_data)
        calculate_bleu(og_data, back_trans_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Back-translation Metrics')
    parser.add_argument('--lan',  nargs='+', type=str, required=True, default='deu', help='a list of desired languages')
    args = parser.parse_args()

    main(args)