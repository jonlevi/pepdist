import random, re
import gzip
from functools import partial
from collections import defaultdict
import os
from itertools import groupby
import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
import matplotlib.pyplot as plt
from numba import jit, njit
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pyrepseq import isvalidaa
from mimetypes import guess_type

aminoacids = 'ACDEFGHIKLMNPQRSTVWY'
aminoacids_set = set(aminoacids)
naminoacids = len(aminoacids)

_aatonumber = {c: i for i, c in enumerate(aminoacids)}
_numbertoaa = {i: c for i, c in enumerate(aminoacids)}

def unique_amino_acids(proteome):
    "returns an array of all unique amino acids used within a proteome"
    return np.unique(list(''.join([seq for h, seq in proteome])))


# Define path variables
repopath = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(repopath, 'data/')

def write_fasta(df, path, seqcolumn=None, idcolumn=None, descriptioncolumn=None):
    """
    Write pandas DataFrame to fasta
    """
    records = []
    for i, row in df.iterrows():
        record = SeqRecord(seq=Seq(row[seqcolumn]),
                           id=row[idcolumn] if idcolumn else '',
                           description=row[descriptioncolumn] if descriptioncolumn else '')
        records.append(record)
    SeqIO.write(records, path, format='fasta')

def make_path(row):
    "Return path based on a row from the proteome file"
    path = row['proteomeid'] + row['shortname'] + '.fasta'
    path += '.gz' if row['speciesid'] else ''
    return path

def load_proteomes(only_pathogens=False):
    """
    Load metadata of proteomes.
    """
    proteomes = pd.read_csv(datadir + 'proteomes.csv', dtype=str, na_filter=False)
    proteomes['path'] = proteomes.apply(make_path, axis=1)
    proteomes.set_index('shortname', inplace=True)
    if only_pathogens:
        mask = proteomes['type'].isin(['bacterium', 'virus', 'parasite'])
        proteomes = proteomes[mask] 
    return proteomes

def proteome_path(name):
    if name == 'ufo':
        return datadir + 'ufos/ufo.fasta'
    if name == 'ext':
        return datadir + 'ufos/ext.fasta'
    if name == 'Humanviruses':
        return datadir+'human-viruses-uniref90-filtered.fasta'
    proteomes = load_proteomes()
    return datadir + proteomes.loc[name]['path']

def fasta_iter(fasta_name, returnheader=True, returndescription=False):
    """
    Given a fasta file return a iterator over tuples of header, complete sequence.
    """
    if returnheader and returndescription:
        raise Exception('one of returnheader/returndescription needs to be False')
    if guess_type(fasta_name)[1] =='gzip':
        _open = partial(gzip.open, mode='rt')
    else:
        _open = open
    with _open(fasta_name) as f:
        fasta_sequences = SeqIO.parse(f, 'fasta')
        for fasta in fasta_sequences:
            if returndescription:
                yield fasta.description, str(fasta.seq)
            elif returnheader:
                yield fasta.id, str(fasta.seq)
            else:
                yield str(fasta.seq)

def map_aatonumber(seq):
    """
    Map sequence to array of number
    """
    seq = np.array(list(seq))
    return np.vectorize(_aatonumber.__getitem__)(seq)

def map_numbertoaa(seq):
    """
    Map integer to amino acid sequence
    """
    seq = list(seq)
    return np.vectorize(_numbertoaa.__getitem__)(seq)


def aatonumber(char):
    return _aatonumber[char]


def map_matrix(matrix, map_=_aatonumber):
    """
    Remap elements in a numpy array 

    Parameters
    ----------
    array : np.array
        Matrix to be remapped
    map_ : dict
        Map to be applied to matrix elements

    Returns
    -------
    np.array
        Remapped matrix
    """
    return np.vectorize(map_.__getitem__)(matrix)

def kmers_to_matrix(kmers):
    """"
    Map a list of str kmers to an integer numpy array.

    Parameters
    ----------
    kmers : iterable of strings
        kmers to be converted
    Returns
    -------
    np.array
        Mapped array
    """
    matrix_str =  np.array([list(kmer) for kmer in kmers])
    matrix = map_matrix(matrix_str)
    return matrix

def matrix_to_kmers(matrix):
    """"
    Map an integer numpy array to a list of str kmers.

    Parameters
    ----------
    matrix: np.array
        Array to be converted
    Returns
    -------
    iterable of strings
        kmers
    """
    char_matrix = map_numbertoaa(matrix)
    kmers = [''.join(row) for row in char_matrix]
    return kmers

def count_kmers(string, k, counter=None, gap=0):
    """
    Count occurrence of kmers in a given string.
    """
    if counter is None:
        counter = defaultdict(int)
    for i in range(len(string)-k-gap+1):
        if gap:
            counter[string[i]+string[i+gap+1:i+k+gap]] += 1
        else:
            counter[string[i:i+k]] += 1
    return counter

def count_kmers_iterable(iterable, k, clean=False, **kwargs):
    """
    Count number of kmers in all strings of an iterable
    """
    counter = defaultdict(int)
    for seq in iterable:
        count_kmers(seq, k, counter=counter, **kwargs)
    if clean:
        counter = {k:counter[k] for k in counter.keys() if isvalidaa(k)}
    return counter

def count_kmers_proteome(proteome, k, **kwargs):
    return count_kmers_iterable(fasta_iter(proteome, returnheader=False), k, **kwargs)


codon_map = {"UUU":"F", "UUC":"F", "UUA":"L", "UUG":"L",
    "UCU":"S", "UCC":"S", "UCA":"S", "UCG":"S",
    "UAU":"Y", "UAC":"Y", "UAA":"STOP", "UAG":"STOP",
    "UGU":"C", "UGC":"C", "UGA":"STOP", "UGG":"W",
    "CUU":"L", "CUC":"L", "CUA":"L", "CUG":"L",
    "CCU":"P", "CCC":"P", "CCA":"P", "CCG":"P",
    "CAU":"H", "CAC":"H", "CAA":"Q", "CAG":"Q",
    "CGU":"R", "CGC":"R", "CGA":"R", "CGG":"R",
    "AUU":"I", "AUC":"I", "AUA":"I", "AUG":"M",
    "ACU":"T", "ACC":"T", "ACA":"T", "ACG":"T",
    "AAU":"N", "AAC":"N", "AAA":"K", "AAG":"K",
    "AGU":"S", "AGC":"S", "AGA":"R", "AGG":"R",
    "GUU":"V", "GUC":"V", "GUA":"V", "GUG":"V",
    "GCU":"A", "GCC":"A", "GCA":"A", "GCG":"A",
    "GAU":"D", "GAC":"D", "GAA":"E", "GAG":"E",
    "GGU":"G", "GGC":"G", "GGA":"G", "GGG":"G",}
nt_to_ind = {
    'A' : 0,
    'C' : 1,
    'G' : 2,
    'U' : 3
    }
def ntfreq_to_aafreq(ntfreq):
    frequencies = {aa:0 for aa in aminoacids}
    for nts, aa in codon_map.items():
        if not aa == 'STOP':
            frequencies[aa] += np.prod([ntfreq[nt_to_ind[nt]] for nt in nts])
    sum_ = sum(frequencies.values())
    for aa in aminoacids:
        frequencies[aa] /= sum_
    return frequencies

def dict_to_array(dict_):
    "return an array from a dictionary by sorting the keys"
    keys = sorted(dict_.keys())
    return np.array([dict_[key] for key in keys])

from operator import ne
def disthamming(seqa, seqb):
    """Calculate Hamming distance between two sequences."""
    return sum(map(ne, seqa, seqb))

@jit(nopython=True)
def hammingdist_jit(seqa, seqb):
    return np.sum(seqa != seqb)
