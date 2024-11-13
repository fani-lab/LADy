"""Unit tests for implicit review loading"""
import sys
import os
import pytest
from ev_implicit_reviews import SEMEVAL_EXPLICIT, SEMEVAL_IMPLICIT, SEMEVAL_BOTH, SEMEVAL_NULL
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from cmn.review import Review
from cmn.semeval import SemEvalReview
from cmn.twitter import TwitterReview # Currently, no twitter reviews have implicit aspects

SEMEVAL_PATH = "./data/raw/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml"

@pytest.mark.parametrize("path, expected", [
    (SEMEVAL_PATH, SEMEVAL_IMPLICIT),
])
def test_implicit(path, expected):
    """Test loading implicit aspect containing reviews."""
    reviews: list[Review] = SemEvalReview.load(path, explicit=False, implicit=True)
    first = reviews[0].to_dict()[0]
    last = reviews[-1].to_dict()[0]
    count = len(reviews)
    assert first == expected["first"]
    assert last == expected["last"]
    assert count == expected["count"]

@pytest.mark.parametrize("path, expected", [
    (SEMEVAL_PATH, SEMEVAL_EXPLICIT),
])
def test_explicit(path, expected):
    """Test loading explicit aspect containing reviews."""
    reviews: list[Review] = SemEvalReview.load(path, explicit=True, implicit=False)
    first = reviews[0].to_dict()[0]
    last = reviews[-1].to_dict()[0]
    count = len(reviews)
    assert first == expected["first"]
    assert last == expected["last"]
    assert count == expected["count"]

@pytest.mark.parametrize("path, expected", [
    (SEMEVAL_PATH, SEMEVAL_BOTH),
])
def test_implicit_and_explicit(path, expected):
    """Test loading both implicit and explicit reviews."""
    reviews: list[Review] = SemEvalReview.load(path, explicit=True, implicit=True)
    first = reviews[0].to_dict()[0]
    last = reviews[-1].to_dict()[0]
    count = len(reviews)
    assert first == expected["first"]
    assert last == expected["last"]
    assert count == expected["count"]

@pytest.mark.parametrize("path, expected", [
    (SEMEVAL_PATH, SEMEVAL_NULL),
])
def test_null(path, expected):
    """Test loading neither implicit nor explicit reviews."""
    reviews: list[Review] = SemEvalReview.load(path, explicit=False, implicit=False)
    count = len(reviews)
    assert count == expected["count"]
