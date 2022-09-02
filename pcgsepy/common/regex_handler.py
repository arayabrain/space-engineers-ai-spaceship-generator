from copy import copy
import re
from typing import Tuple
from functools import total_ordering


@total_ordering
class MyMatch:
    def __init__(self,
                 lhs: str,
                 span: Tuple[int, int],
                 lhs_string: str) -> None:
        self.lhs = lhs
        self.start = span[0]
        self.end = span[1]
        self.lhs_string = lhs_string

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, MyMatch) and self.start == __o.start and self.end == __o.end
        
    def __lt__(self, __o: object) -> bool:
        return isinstance(__o, MyMatch) and (self.start < __o.start or (self.start == __o.start and self.end > __o.end))
    
    def __str__(self) -> str:
        return f'{self.lhs=}\n\tself.span={self.start},{self.end}\n\t{self.lhs_string=}'
    
    def __repr__(self) -> str:
        return str(self)


char_to_re = {
    '(': '\(',
    ')': '\)',
    '[': '\[',
    ']': '\]',
    'x': '\d',
    'X': '\d',
    'Y': '\d',
}


def extract_regex(lhs: str) -> str:
    """Extract the regex from the LHS rule.

    Args:
        lhs (str): The LHS rule (human-readable).

    Returns:
        str: The compiled regex.
    """
    r = copy(lhs)
    for k, v in char_to_re.items():
        r = r.replace(k, v)
    return re.compile(r)