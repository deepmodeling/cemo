from argparse import _SubParsersAction as SPA
from ._all_parsers import all_parsers
from functools import reduce
from collections.abc import Callable
import sys
from typing import TYPE_CHECKING
if sys.version_info >= (3, 9) or TYPE_CHECKING:
    FnParserType = Callable[[SPA], SPA]
else:
    FnParserType = Callable


def add_all_parsers(subparsers: SPA) -> SPA:
    def add_parser(acc: SPA, f: FnParserType) -> SPA:
        # The left argument, acc, is the accumulated value and 
        # the right argument, parse, is the update value from the iterable.
        return f(acc)

    return reduce(add_parser, all_parsers(), subparsers)
