import argparse
from functools import reduce
from typing import Callable, List
ArgParser = argparse.ArgumentParser
FnParser = Callable[[ArgParser], ArgParser]


def add_parsers(base_parser: ArgParser, parsers: List[FnParser]) -> ArgParser:
    """
    Add additional parsers to the parser object.

    Args:
        base_parser: a base parser object.
        parsers: A list of parser functions.

    Returns:
        The parser object with the additional parsers added.
    """
    return reduce(lambda p, fn: fn(p), parsers, base_parser)
