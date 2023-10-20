import os
from aml.fast import Fast

from typing import List, Tuple
import pandas as pd

from nltk.corpus import wordnet as wn

import params
from cmn.review import Review
from cmn.semeval import SemEvalReview

data = "../output/toy.2016SB5/reviews.pkl"
output = "../output/toy.2016SB5/fast/"

reviews = pd.read_pickle(data)
if not os.path.isdir(output): os.makedirs(output)

am = Fast(naspects=5, nwords=params.settings['train']['nwords'])
am.train(reviews, [], params.settings['train']['fast'], params.settings['prep']['doctype'], None, output)



