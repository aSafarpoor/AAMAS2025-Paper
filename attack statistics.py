import heapq
import numpy as np
from tqdm import tqdm
import random
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import os 
import json 
import time

# Function to measure elapsed time between start and end
def timer(start_time, end_time):
    elapsed_time_s = end_time - start_time
    elapsed_time_ms = elapsed_time_s * 1000
    print(f"Elapsed time in milliseconds: {elapsed_time_ms:.6f} ms")

# Function to load a text file, assuming it contains integers, and return a list of integers
def load_txt_file(filename):
    with open(filename, 'r') as file:
        return [int(line.strip()) for line in file]

# Function to load a text file, assuming each line contains edges (pairs of integers), and return a list of lists of integers
def load_txt_file_for_edges(filename):
    with open(filename, 'r') as file:
        return [list(map(int, line.strip().split())) for line in file]

# Function to load a JSON file into a dictionary, with keys converted to integers and values to floats
def load_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        return {int(float(k)): float(v) for k, v in data.items()}

# Iterate over different dataset names
for name1 in ['Pokec+', 'twitter+', 'facebook+', 'lastfm+']:
    for name2 in ['random', 'ba', 'bfs']:  # Types of graph structures (random, ba, bfs)

        # Build the dataset path using the names
        name = f'./datasets/{name1}{name2}/'
        print("\n\n", name1, name2)

        # Load edges, benign nodes, and sybil nodes from their respective files
        edges = load_txt_file_for_edges(name + 'edges.txt')
        benigns = load_txt_file(name + 'benigns.txt')
        sybils = load_txt_file(name + 'sybils.txt')

        # Flatten the list of edges into a list of unique nodes
        nodes = list(set(np.array(edges).reshape(-1)))

        # Load dictionaries for resistance and probability of resistance from JSON files
        r = load_json_file(os.path.join(name, 'resistanceDictionary.json'))
        pr = load_json_file(os.path.join(name, 'probability_of_resistance_dictionary.json'))

        # Calculate the average resistance and probability of resistance for benign nodes
        rstar = 0
        prstar = 0
        for node in benigns:
            rstar += r[node]
            prstar += pr[node]
        print("|edges|= ", len(edges))  # Print the number of edges

        # Print average resistance and probability of resistance for benign nodes
        print('r*', 'pr*', rstar/len(benigns), prstar/len(benigns))

        # Convert sybil nodes to a set for faster lookup
        sybils = set(sybils)

        # Initialize counters for various types of interactions
        attacks = 0
        reverseofattacks = 0
        b2b = 0  # Benign-to-benign edges
        s2s = 0  # Sybil-to-sybil edges

        # Loop over each edge to count different types of interactions
        for edge in edges:
            h = edge[0]  # Head of the edge
            t = edge[1]  # Tail of the edge

            if h in sybils:  # If the head is a sybil
                if t in sybils:  # If the tail is also a sybil
                    s2s += 1  # Sybil-to-sybil edge
                else:
                    attacks += 1  # Sybil-to-benign edge (attack)
            else:
                if t in sybils:  # If the tail is a sybil
                    reverseofattacks += 1  # Benign-to-sybil edge (reverse of attack)
                else:
                    b2b += 1  # Benign-to-benign edge

        # Print the count of various interactions
        print("attacks: {}, reverse of attacks: {}, s2s: {}, b2b: {}".format(attacks, reverseofattacks, s2s, b2b))

        # Calculate and print average in-degree and out-degree for sybils and benign nodes
        print('avg d_in sybils: ', (s2s + reverseofattacks) / len(sybils))
        print('avg d_out sybils: ', (s2s + attacks) / len(sybils))
        print('avg d_in benigns: ', (b2b + attacks) / len(benigns))
        print('avg d_out benigns: ', (b2b + reverseofattacks) / len(benigns))
