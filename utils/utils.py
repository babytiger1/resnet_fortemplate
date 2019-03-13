import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='../configs/example.json',
        help='The Configuration file')
    args = argparser.parse_args()
    print(args)
    return args
