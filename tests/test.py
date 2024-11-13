import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from cmn.review import Review
from cmn.semeval import SemEvalReview
from cmn.twitter import TwitterReview

reviews = SemEvalReview.load("./data/raw/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml", explicit=False, implicit=True)
print(len(reviews))
for review in reviews:
    print(review.to_dict()[0])
