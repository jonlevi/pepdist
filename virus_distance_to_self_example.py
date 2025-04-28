import numpy as np
import pandas as pd
from utils import count_kmers_proteome, proteome_path
import sys

from distances import nearest_cross_reactivity_distance_parallel, nearest_hamming_parallel

human = proteome_path('Human')
human_reference = set(count_kmers_proteome(human, 9, clean=True))



if __name__ == "__main__":
        viruses = set(count_kmers_proteome(proteome_path('Humanviruses'), 9, clean=True))
        input_kmers = viruses
        cross_reactivity_distances = nearest_cross_reactivity_distance_parallel(input_kmers,human_reference)

        hamming_distances = nearest_hamming_parallel(input_kmers, human_reference)

        rows = [{'seq': s, 'nearest_hamming': hamming_distances[s], 'nearest_cross_reactivity_distance': cross_reactivity_distances[s]} for s in input_kmers]

        pd.DataFrame(rows).to_csv('output/no_sampling_virus_example.csv')

        