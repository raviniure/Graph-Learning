import os
import pickle
from argparse import ArgumentParser
import imdb as imdb
import HIV as hiv
import Mutagenicity as muta
import Amazon as amz


def is_valid_file(parser, arg):
    """
    :param parser: parser object that was initialized
    :param arg: io.TextIOWrapper
    :return: return an open file handle
    """
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'rb')  # return an open file handle


if __name__ == "__main__":
    """
    Command liner parser initialization
    """
    parser = ArgumentParser(description="Path name for training and evaluation dataset")

    parser.add_argument("--dataset", dest="dataset", required=True,
                        help="Training Dataset", metavar="STRING")

    parser.add_argument("--eval_path", dest="eval_path", required=True,
                        help="Evaluation Dataset", metavar="FILE", type=lambda x: is_valid_file(parser, x))

    args = parser.parse_args()

    """
    Load dataset
    """
    if args.dataset == 'IMDB':
        train_file_path = './datasets/IMDB/IMDB_Train/data.pkl'
        with open(train_file_path, 'rb') as f:
            train = pickle.load(f)
        evaluation = pickle.load(args.eval_path)

        imdb.train(train, evaluation)

    elif args.dataset == 'HIV':
        train_file_path = './datasets/HIV/HIV_Train/data.pkl'
        with open(train_file_path, 'rb') as f:
            train = pickle.load(f)
        evaluation = pickle.load(args.eval_path)

        hiv.train(train, evaluation)

    elif args.dataset == 'Mutagenicity':
        train_file_path = './datasets/Mutagenicity/Mutagenicity_Train/data.pkl'
        with open(train_file_path, 'rb') as f:
            train = pickle.load(f)
        evaluation = pickle.load(args.eval_path)

        muta.train(train, evaluation)

    elif args.dataset == 'Amazon':
        train_file_path = './datasets/Amazon/Amazon_Train/data.pkl'
        with open(train_file_path, 'rb') as f:
            train = pickle.load(f)
        evaluation = pickle.load(args.eval_path)

        amz.train(train, evaluation)

    else:
        print("Error: Wrong dataset provided")


