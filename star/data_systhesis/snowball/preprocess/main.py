import argparse
from utils import *

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True, help='dataset type, ie. spider')
    parser.add_argument('--input', type=str, required=True, help='input dir')
    parser.add_argument('--output', type=str, required=True, help='output dir')
    print(parser.parse_args())
    return parser.parse_args()

def main():
    args = get_arg()
    mkdir(args.output)
    create_output(args.type, args.input, args.output)

main()
