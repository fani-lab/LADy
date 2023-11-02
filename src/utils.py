import operator
import functools
from typing import Callable, List, TypeVar, Optional
from more_itertools import first_true
from returns import maybe, pipeline
from returns.pipeline import pipe
from returns.curry import curry

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

def find_first(xs: List[T], compare: Callable[[T], bool]) -> Optional[T]:  return first_true(xs, default=None, pred=compare)
