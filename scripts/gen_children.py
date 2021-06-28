#! /bin/python
import os,sys
path = os.path.dirname(os.getcwd())
sys.path.append(path)

import numpy as np
from Bio import SeqIO
import argparse
import datetime
import yaml
from utils import models

from utils import preprocessing as pre

parser = argparse.ArgumentParser(""" """)
parser.add_argument("-i", "--input", type=str, default = None)
parser.add_argument("-w", "--weights", type=str, default = None)
parser.add_argument("-c", "--config", type=str, default = None)
parser.add_argument('-o', '--output', type=str, default = None,
                   help = 'Name to store data')
parser.add_argument('-n', '--num_data_children', type=int, default=32,
                   help = "Number of Children generated per parent")
parser.add_argument('-v', '--verbose', action="store_true",
                   help = "Verbosity")
parser.add_argument("-g", "--gpu", type=str, default = '1')


VOCAB_SIZE = 21
AA = "ACDEFGHIKLMNPQRSTVWYX"
AA_DICT = {index: aa for index, aa in enumerate(AA)}

FASTA_STR = ">parent_{}_child_{} \n{}\n"


def writing_fasta(children, name = "data"):
    with open(name, "w") as f:
        for p, key in enumerate(children.keys()):
            for i, child in enumerate(children[key]):
                #string = "".join([AA_DICT[key] for key in child])
                f.write(FASTA_STR.format(str(p), str(i), child))


def main(args):
    experiments = os.listdir("../results")
    experiments.sort(key=lambda date: datetime.datetime.strptime(date, "%Y%m%d-%H%M%S"))
    latest_experiment = experiments[-1]
    if args.config == None:
        args.config = os.path.join("../results", latest_experiment, "config.yaml")
    if args.weights == None:
        args.weights = os.path.join("../results",latest_experiment, "weights/virus_gan_model")
    if args.output == None:
        args.output = os.path.join("../results", latest_experiment, "gen_children.fasta")

    # Get time stamp
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Load configuration file
    with open(args.config, 'r') as file_descriptor:
        config = yaml.load(file_descriptor, Loader=yaml.FullLoader)

    # Load test data
    file_parents    = config["Data"]['file_parents_test']
    file_children   = config["Data"]['file_children_test']
    seq_length      = config["Data"]['seq_length']
    max_samples     = config["Data"]['max_samples']

    data = pre.prepare_dataset(file_parents, file_children, seq_length = seq_length, max_samples = max_samples, training = False)
    matched_sequences = pre.get_matched_parent_and_child_sequences(file_parents,
                                                                   file_children,
                                                                   max_seq_length = seq_length,
                                                                   max_num_of_sequences = max_samples)
    parent_ids = matched_sequences['parent_id'].values

    # Initiate model
    model = models.VirusGan(config)
    model.load_weights(args.weights)

    if args.verbose:
        print("Generating children")
    generated_children = model.generate(data,
                                        n_children = args.num_data_children,
                                        parent_ids = parent_ids)

    if args.verbose:
        print("Writing fasta")

    writing_fasta(generated_children, name = args.output)

    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    main(args)
