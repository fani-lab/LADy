import operator
import functools
from typing import Callable, List, TypeVar
from more_itertools import first_true
from returns.maybe import Maybe

T = TypeVar('T')

def remove_duplicates_from_list(xs: List[T], compare: Callable[[T, T], bool]) -> List[T]:
    """Removes duplicates from a list

    Args:
        xs (List[T])
        compare (Callable[[T, T], bool]): compare function to determine if two elements are equal

    Returns:
        A new List with no duplicates
    """
    result = []
    for d in xs:
        if not any(compare(d, r) for r in result):
            result.append(d)
    return result

def flatten(xs: List[List[T]] ) -> List[T]:
    """Assume we have a list of list of T, this function will flatten it to a list of T

    Args:
        xs (List[List[T]])

    Returns:
        List[T]: Flattened
    """
    return functools.reduce(operator.iconcat, xs, [])

def find_first(xs: List[T], prediction: Callable[[T], bool]) -> Maybe[T]:
    """ Find first occurrence in the list of T by prediction function 

    Args: 
        xs (List[T])
        prediction: determine if the element is the one that we want given an item and returns boolean


    Return:
        Maybe[T]: maybe found or not
    """
    return Maybe.from_optional(first_true(xs, default=None, pred=prediction))

def raise_exception_fn(exception: str):
    raise Exception(exception)
