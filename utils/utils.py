import argparse


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("-m", "--mode", choices=['train', 'test'],
                            required=True,
                            help="Choose the mode, train or test.")
    arg_parser.add_argument("-d", "--device", choices=['cpu', 'cuda:0'],
                            default='cpu', help="Select the running device.")
    arg_parser.add_argument("-s", "--save_models", action='store_true',
                            default=True, help="Whether save the tuned model.")

    return arg_parser
