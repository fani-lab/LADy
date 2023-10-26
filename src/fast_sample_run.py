import os

import numpy as np
from aml.fast import Fast

from typing import List, Tuple
import pandas as pd

from nltk.corpus import wordnet as wn

import params
from cmn.review import Review
from cmn.semeval import SemEvalReview

from main import split

data = "../output/toy.2016SB5/reviews.pkl"
output = "../output/toy.2016SB5/fast/"

if not os.path.isdir(output): os.makedirs(output)
reviews = pd.read_pickle(data)
splits = split(len(reviews), output)

am = Fast(naspects=5, nwords=params.settings['train']['nwords'])

for f in splits['folds'].keys():
    reviews_train = np.array(reviews)[splits['folds'][f]['train']].tolist()
    reviews_valid = np.array(reviews)[splits['folds'][f]['valid']].tolist()
    am.train(reviews_train, reviews_valid, params.settings['train']['fast'], 
             params.settings['prep']['doctype'], params.settings['train']['no_extremes'], f'{output}/f{f}.')
    
# simple test for inferring using fasttext model
for f in splits['folds'].keys():
    reviews_test = np.array(reviews)[splits['test']].tolist()
    for r in reviews_test:
        r_pred_aspect = am.infer(r, doctype=params.settings['prep']['doctype'])
        print(r_pred_aspect)



