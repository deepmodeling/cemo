import argparse
from cemo.tk.parser.add_parsers import add_parsers


def test_add_parsers():
    # Define the base parser
    base_parser = argparse.ArgumentParser()

    # Define the additional parsers
    def parser1(p):
        p.add_argument('--arg1')
        return p

    def parser2(p):
        p.add_argument('--arg2')
        return p

    parsers = [parser1, parser2]

    # Call the function with the parameters
    parser = add_parsers(base_parser, parsers)

    # Check if the arguments were added to the parser
    args = parser.parse_args(['--arg1', 'value1', '--arg2', 'value2'])
    assert args.arg1 == 'value1'
    assert args.arg2 == 'value2'
