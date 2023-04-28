#! /bin/python

import numpy as np
from Bio import SeqIO
import argparse

parser = argparse.ArgumentParser(""" """)

parser.add_argument('-o', '--output_file', type=str, default = 'data',
                   help = 'Name to store data')
parser.add_argument('-n', '--num_data_points', type=int, default=1000,
                   help = "Number of data points")
parser.add_argument('-l', '--sequence_length', type=int, default=100,
                   help = "length of sequences generated")

parser.add_argument('-v', '--verbose', action="store_true",
                   help = "Verbosity")


args = parser.parse_args()

VOCAB_SIZE = 20

AA = "ARNDCQEGHILKMFPSTWYVBZX"
AA_DICT = {index: aa for index, aa in enumerate(AA)}

FASTA_STR = ">{} \n{}\n"


def generate_parents(num = 1000, seq_length = 100):
    parents = np.random.randint(VOCAB_SIZE, size=[num,seq_length])
    return parents

def generate_children(parents, num_children=1, seq_length= 100, n_mutations= 5):
    children = parents
    for i in range(children.shape[0]):
        mutation_indices = np.random.randint(seq_length, size = n_mutations)
        mutations = np.random.randint(VOCAB_SIZE, size = n_mutations)
        children[i,mutation_indices] = mutations
    return children

def writing_fasta(parents, children, name = "data"):
    with open(name+"-parents.fasta", "w") as f:
        for i in range(parents.shape[0]):
            f.write(FASTA_STR.format(str(i), "".join([AA_DICT[key] for key in parents[i,:]])))
    with open(name+"-children.fasta", "w") as f:
        for i in range(parents.shape[0]):
            f.write(FASTA_STR.format(str(i), "".join([AA_DICT[key] for key in children[i,:]])))
def main(args):
    
    if args.verbose:
        print("Making parents")
    
    parents  = generate_parents(num = args.num_data_points, seq_length = args.sequence_length)
    
    if args.verbose:
        print("Making children")
    children = generate_children(parents, seq_length= args.sequence_length)
    
    if args.verbose:
        print("Writing fasta")
    writing_fasta(parents, children, name = args.output_file)    
    
    return 0

if __name__ == "__main__":
    main(args)