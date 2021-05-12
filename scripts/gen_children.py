#! /bin/python
import os,sys
path = os.path.dirname(os.getcwd())
sys.path.append(path)

import numpy as np
from Bio import SeqIO
import argparse

from utils import preprocessing as pre

parser = argparse.ArgumentParser(""" """)

parser.add_argument("-i", "--input", type=str, default = '../data/parents')

parser.add_argument("-w", "--weights", type=str, default = None)

parser.add_argument("-c", "--config", type=str, default = "../config/config.yaml")

parser.add_argument('-o', '--output', type=str, default = '../data/gen.children.fasta',
                   help = 'Name to store data')
parser.add_argument('-n', '--num_data_children', type=int, default=1000,
                   help = "Number of data points")


parser.add_argument('-v', '--verbose', action="store_true",
                   help = "Verbosity")




VOCAB_SIZE = 21

AA = "ARNDCQEGHILKMFPSTWYVBZX"
AA_DICT = {index: aa for index, aa in enumerate(AA)}

FASTA_STR = ">{} \n{}\n"


def generate_children(parents, num_children=1, seq_length= 100, n_mutations= 5):
    children = parents
    for i in range(children.shape[0]):
        mutation_indices = np.random.randint(seq_length, size = n_mutations)
        mutations = np.random.randint(VOCAB_SIZE, size = n_mutations)
        children[i,mutation_indices] = mutations
    return children

def writing_fasta(children, name = "data"):
    with open(name+"-children.fasta", "w") as f:
        for i, child in enumerate(children):
            f.write(FASTA_STR.format(str(i), "".join([AA_DICT[key] for key in child)))
                                                      
def main(args):
    
    if args.verbose:
        print("Making parents")
    
    
    
    # Get time stamp
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Load configuration file
    with open(args.config, 'r') as file_descriptor:
        config = yaml.load(file_descriptor, Loader=yaml.FullLoader)
        
    with open(args.config, 'r') as file_descriptor:
        config_str = file_descriptor.read()

    # Load training data
    file_parents    = config['file_parents']
    file_children   = config['file_children']
    seq_length      = config['seq_length']
    max_samples     = config['max_samples']
    data = pre.prepare_data(file_parents, file_children, seq_length = seq_length, max_samples = max_samples)
    
    # Initiate model
    model = models.VirusGan(config)
    model.load_weights(args.weights)
    
    
    if args.verbose:
        print("Generating children")
        generated_children = model.generate(data)
    
    if args.verbose:
        print("Writing fasta")
        
    writing_fasta(parents, children, name = args.output_file)    
    
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    main(args)
