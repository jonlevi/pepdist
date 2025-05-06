import numpy as np
import pandas as pd
from utils import count_kmers_proteome, proteome_path, kmers_to_matrix
import sys
import argparse
import random
import os
from tqdm import tqdm
from distances import get_hamming_distance_from_set, get_cross_reactivity_distance_from_set, get_hamming_distance_from_set_alt

DISTANCES = {'hamming': get_hamming_distance_from_set_alt, 'cross_reactivity': get_cross_reactivity_distance_from_set}

human = proteome_path('Human')
human_reference = count_kmers_proteome(human, 9, clean=True)


def compute_distances(input_kmers, reference, distance_metric_key, aggregator):

        if distance_metric_key not in DISTANCES:
                raise Exception(f"{distance_metric_key} not one of {DISTANCES.keys()}")

        distance_function  = DISTANCES[distance_metric_key]
        rows=[]
        for kmer in tqdm(input_kmers):
                rows.append({'seq': kmer, 'distance': distance_function(kmer, reference)})
        return rows

if __name__ == "__main__":
        
        parser = argparse.ArgumentParser()

        # parser.add_argument("-input_path")

        # should be a key that can be passed into proteome_path, see data directory
        parser.add_argument("--proteome")
        parser.add_argument("--N")
        parser.add_argument("--iter")
        parser.add_argument("--metric")
        parser.add_argument("--aggregator")

        args = parser.parse_args()

        N_iter = 1 if args.iter is None else int(args.iter)

        input_kmers = set(count_kmers_proteome(proteome_path(args.proteome), 9, clean=True))

        for i in range(N_iter):
                path = f'output/{args.proteome}_{args.aggregator}_distance_to_self'
                if args.N is not None:
                        path = path + f"_{args.N}"
                else:
                         path = path + f"_no_sampling"
                if N_iter > 1:
                        path = path + f"_iter_{i}"
                
                path = f'{path}.csv'

                if os.path.exists(path):
                        continue

                print(i)
                if args.N is not None:
                        n = int(args.N)
                        # sample without replacement
                        individuals = [identity for identity, count in human_reference.items() for _ in range(count)]
                        sampled = random.sample(individuals, n)
                        reference_set = set(sampled)

                else:
                         reference_set = set(human_reference)

                ref_matrix = kmers_to_matrix(reference_set)

                distances = compute_distances(input_kmers,ref_matrix,args.metric, args.aggregator)

                pd.DataFrame(distances).to_csv(f'{path}.csv')

        