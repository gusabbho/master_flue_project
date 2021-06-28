import numpy as np
import tensorflow as tf
from Bio import SeqIO
from sklearn.model_selection import train_test_split
import pandas as pd


def convert_table(seq):
    aas = 'ACDEFGHIKLMNPQRSTVWYX'
    dict_ = {i: aa for i, aa in enumerate(aas)}
    seq_str = "".join([dict_[res] for res in list(seq)])
    return seq_str


def to_binary(seq, max_length, start_stop=False):
    # eoncode non-standard amino acids like X as all zeros
    # output a array with size of L*20
    seq = seq.upper()
    if not start_stop:
        aas = 'ACDEFGHIKLMNPQRSTVWYX'
        vocab = 21
    else:
        aas = 'ACDEFGHIKLMNPQRSTVWYX<>'
        vocab = 23
    pos = dict()
    for i in range(len(aas)): pos[aas[i]] = i

    binary_code = dict()
    for aa in aas:
        code = np.zeros(vocab, dtype=np.float32)
        code[pos[aa]] = 1
        binary_code[aa] = code

    seq_coding = np.zeros((max_length, vocab), dtype=np.float32)
    for i, aa in enumerate(seq):
        code = binary_code.get(aa, np.zeros(vocab, dtype=np.float32))
        seq_coding[i, :] = code
    return seq_coding


def zero_padding(inp, length=500, start=False):
    # zero pad input one hot matrix to desired length
    # start .. boolean if pad start of sequence (True) or end (False)
    # assert len(inp) <= length
    out = np.zeros((length, inp.shape[1]))
    if start:
        out[-inp.shape[0]:] = inp[:length, :]
    else:
        out[0:inp.shape[0]] = inp[:length, :]
    return out


def loss_weight(mask, max_length):
    len_seq = len(mask)
    seq_w = [1 for i in mask]
    tmp = np.ones((max_length,))
    tmp[:len_seq] = seq_w
    tmp[len_seq:] = 0.0
    return tmp


def prepare_dataset(file_parents,
                    file_children,
                    seq_length=1024,
                    t_v_split=0.1,
                    max_samples=5000,
                    training=True):
    max_seq_length = seq_length
    max_num_of_sequences = max_samples

    matched_sequences = get_matched_parent_and_child_sequences(file_parents, file_children, max_seq_length, max_num_of_sequences)
    parents_seq = matched_sequences['parent_sequence']
    children_seq = matched_sequences['child_sequence']

    # bin = one-hot encoded sequences
    parents_seq_bin = []
    children_seq_bin = []
    for seq in parents_seq.values:
        parents_seq_bin.append(to_binary(seq, max_length=max_seq_length))
    for seq in children_seq.values:
        children_seq_bin.append(to_binary(seq, max_length=max_seq_length))

    # DEBUG:
    if not training:
        from IPython.display import display
        print('=' * 80)
        display(parents_seq)
        display(children_seq)
        print(len(parents_seq_bin))
        print(len(children_seq_bin))
        print('=' * 80)
        print()

    # Splitting data to training and validation sets
    if training:
        parent_train, parent_test, child_train, child_test = train_test_split(
            np.array(parents_seq_bin, dtype=np.float32),
            np.array(children_seq_bin, dtype=np.float32),
            test_size=t_v_split, random_state=42)

        dataset_train = tf.data.Dataset.from_tensor_slices((parent_train, child_train))
        dataset_validate = tf.data.Dataset.from_tensor_slices((parent_test, child_test))
        return dataset_train, dataset_validate

    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            (np.array(parents_seq_bin, dtype=np.float32),
             np.array(children_seq_bin, dtype=np.float32)))
        return dataset


def get_matched_parent_and_child_sequences(file_parents, file_children, max_seq_length, max_num_of_sequences):
    parent_child_association_table = pd.read_csv(
        '/home/hosseina/codes/data_preparation/MSA_16.3/phylo_by_RAxML/ML/master_flue_project/scripts/parent_child_association.csv')
    parent_child_association_table = parent_child_association_table[['parent_id', 'child_id']]

    parent_sequences = fasta_to_seq_df(file_parents, ['parent_id', 'parent_sequence'])
    child_sequences = fasta_to_seq_df(file_children, ['child_id', 'child_sequence'])

    print("#" * 80)
    print('N test children before matching:', child_sequences.shape[0])
    print("#" * 80)

    # columns: parent_id, child_id, parent_seq
    parent_seq_with_child_ids = pd.merge(
        parent_child_association_table,
        parent_sequences,
        on='parent_id'
    )

    # parent_id, child_id, parent_seq, child_seq
    matched_sequences = pd.merge(
        parent_seq_with_child_ids,
        child_sequences,
        on='child_id'
    )

    print("#" * 80)
    print('N unique test children after matching:', matched_sequences['child_id'].unique().shape[0])
    print("#" * 80)

    matched_sequences = matched_sequences.drop_duplicates(subset=['child_id'])
    print("#" * 80)
    print('N test children after fixing:', matched_sequences['child_id'].shape[0])
    print("#" * 80)

    # Remove child and parent sequences longer than max_seq_length
    matched_sequences = matched_sequences[
        (matched_sequences['parent_sequence'].str.len() <= max_seq_length)
        & (matched_sequences['child_sequence'].str.len() <= max_seq_length)
        ]

    # Keep only first max_num_of_sequences sequences
    if matched_sequences.shape[0] > max_num_of_sequences:
        matched_sequences = matched_sequences.head(max_num_of_sequences)

    return matched_sequences


def fasta_to_seq_df(filename: str, column_names: list) -> pd.DataFrame:
    return pd.DataFrame([[entry.id, str(entry.seq)]
                         for entry in
                         list(SeqIO.parse(str(filename), 'fasta'))],
                        columns=column_names)


def save_df_as_fasta(df, id_col, seq_col, fasta_filename):
    df = df[[id_col, seq_col]]
    entries = []
    for row in df.values:
        record = SeqRecord(Seq(row[1]), id=row[0], name=row[0], description='')
        entries.append(record)
    with open(fasta_filename, 'w') as outfile:
        SeqIO.write(entries, outfile, 'fasta')
