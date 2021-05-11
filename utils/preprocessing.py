import os
import numpy as np
import tensorflow as tf
from Bio import SeqIO
from sklearn.model_selection import train_test_split
# TODO: Implement data loader

def to_binary(seq, max_length, start_stop = False):
    # eoncode non-standard amino acids like X as all zeros
    # output a array with size of L*20
    seq = seq.upper()
    if not start_stop:
        aas = 'ACDEFGHIKLMNPQRSTVWYX'
        vocab=21
    else:
        aas = 'ACDEFGHIKLMNPQRSTVWYX<>'
        vocab=23
    pos = dict()
    for i in range(len(aas)): pos[aas[i]] = i
    
    binary_code = dict()
    for aa in aas: 
        code = np.zeros(vocab, dtype = np.float32)
        code[pos[aa]] = 1
        binary_code[aa] = code
    
    seq_coding = np.zeros((max_length,vocab), dtype = np.float32)
    for i,aa in enumerate(seq): 
        code = binary_code.get(aa,np.zeros(vocab, dtype = np.float32))
        seq_coding[i,:] = code
    return seq_coding

def zero_padding(inp,length=500,start=False):
    # zero pad input one hot matrix to desired length
    # start .. boolean if pad start of sequence (True) or end (False)
    #assert len(inp) <= length
    out = np.zeros((length,inp.shape[1]))
    if start:
        out[-inp.shape[0]:] = inp[:length,:]
    else:
        out[0:inp.shape[0]] = inp[:length,:]
    return out

def loss_weight(mask, max_length):
    len_seq = len(mask)
    seq_w = [1 for i in mask] 
    tmp = np.ones((max_length,))
    tmp[:len_seq] = seq_w
    tmp[len_seq:] = 0.0
    return tmp

def prepare_dataset(file_parents, file_children , seq_length = 1024, t_v_split = 0.1, max_samples = 5000):
    
  

    count=0
    dict_parents = {'id':[] ,'mask':[],'seq':[], 'seq_bin':[], 'loss_weight':[]}
    dict_children = {'id':[] ,'mask':[],'seq':[], 'seq_bin':[], 'loss_weight':[]}
    # loading data to dict
    for i, rec in enumerate(SeqIO.parse(file_parents, 'fasta')):
        count +=1
        if count >max_samples:
            break
        if len(rec.seq)>seq_length:
            continue
        dict_parents['id'].append(rec.id)
        dict_parents['seq'].append(str(rec.seq))
        dict_parents['seq_bin'].append(to_binary(rec.seq, max_length=seq_length))
    count=0
    for i, rec in enumerate(SeqIO.parse(file_children, 'fasta')):
        count +=1
        if count >max_samples:
            break
        if len(rec.seq)>seq_length:
            continue
        dict_children['id'].append(rec.id)
        dict_children['seq'].append(str(rec.seq))
        dict_children['seq_bin'].append(to_binary(rec.seq, max_length=seq_length))
      
   # Splitting data to training and validation sets
    print(len(dict_parents["seq_bin"]))
    print(len(dict_children["seq_bin"]))
    parent_train, parent_test, child_train, child_test = train_test_split(
                                                    np.array(dict_parents['seq_bin'],dtype=np.float32),
                                                    np.array(dict_children['seq_bin'], dtype = np.float32),
                                                    test_size=t_v_split, random_state=42)

    dataset_train = tf.data.Dataset.from_tensor_slices((parent_train, child_train))
    dataset_validate = tf.data.Dataset.from_tensor_slices((parent_test, child_test))
    return dataset_train, dataset_validate