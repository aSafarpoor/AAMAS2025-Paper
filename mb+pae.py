import heapq
import numpy as np
from tqdm import tqdm
import random
import math
import matplotlib.pyplot as plt
import json
import os

import time
def timer(start_time,end_time):
    elapsed_time_s = end_time - start_time
    elapsed_time_ms = elapsed_time_s * 1000
    print(f"Elapsed time in milliseconds: {elapsed_time_ms:.6f} ms")

# Function to load text file into a list
def load_txt_file(filename):
    with open(filename, 'r') as file:
        return [int(line.strip()) for line in file]

# Function to load text file with edges into a list of lists
def load_txt_file_for_edges(filename):
    with open(filename, 'r') as file:
        return [list(map(int, line.strip().split())) for line in file]

# Function to load JSON file into a dictionary
def load_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        return {int(float(k)): float(v) for k, v in data.items()}


def creat_in_degree_neigs_minus_S_B(edges,nodes,sybils,benigns):
    degree_in = {}
    in_neigs = {}
    out_neigs = {}

    for node in nodes:
        degree_in[node] = 0
        in_neigs[node] = []
        out_neigs[node] = []


    union_set = set(sybils).union(set(benigns))

    for edge in tqdm(edges):
        out_neigs[edge[0]].append(edge[1])
        if edge[0] not in union_set:
            degree_in[edge[1]] += 1
            in_neigs[edge[1]].append(edge[0])

    return degree_in,in_neigs,out_neigs




def traversing_resistance_and_degree_aware(nodes,edges,benigns,pr,r,k,degree_in,in_neigs_minus_S_B,out_neigs):
    A = []
    N = benigns[:]

    gammain = {key: value.copy() for key, value in in_neigs_minus_S_B.items()}
    degree = degree_in.copy()
    out_n = {key: value.copy() for key, value in out_neigs.items()}

    while len(A)<k:
        maxnode = -1
        maxval = -1
        for node in N:
            value = pr[node]*degree[node]
            if value>maxval:
                maxval = value
                maxnode = node
        A.append(maxnode)
        N.remove(maxnode)

        v = maxnode

        if r[v]==1:
            for u in gammain[v]:
                N.append(u)

                for w in out_n[u]:
                    gammain[w].remove(u)
                    degree[w]-=1

    return A


def run_bfs_discovery(B, A, r, in_neigs):
    # Convert B, A, and S to sets if they are not already
    B = set(B)
    A = set(A)


    discovered = set()
    visited = set()

    # Process nodes in A that are also in B
    for v in A & B:
        bfs_queue = []
        if r[v] == 1:  # Check if r(v) == 1
            bfs_queue.append(v)

        while bfs_queue:
            w = bfs_queue.pop(0)
            visited.add(w)

            # Add neighbors of w that are not in B ∩ S to discovered set
            discovered.update(set(in_neigs[w]) - (B))

            # For neighbors in A that are not visited yet
            for u in (set(in_neigs[w]) & A) - visited:
                if r[u] == 1:  # Check if r(u) == 1
                    bfs_queue.append(u)

    return len(discovered)



# Function to create in-degree and in-neighbors dictionaries
def create_in_degree_neigs(edges, nodes, benigns, sybils):
    degree_in = {node: 0 for node in nodes}
    in_neigs = {node: [] for node in nodes}

    set_bands = set(benigns).union(set(sybils))

    for edge in edges:
        if edge[0] not in set_bands:
            degree_in[edge[1]] += 1
            in_neigs[edge[1]].append(edge[0])

    return degree_in, in_neigs

# Function to find the top k nodes by sum
def top_k_nodes_sum(d, k):
    top_k = heapq.nlargest(k, d.items(), key=lambda x: x[1])
    A = [key for key, value in top_k]
    top_k_sum = sum(value for key, value in top_k)
    return A, top_k_sum



for name1 in ['twitter+','Pokec+','facebook+','lastfm+']:
    for name2 in ['random', 'ba', 'bfs']:
        name = f'datasets/{name1}{name2}/'
        
        print(name)

        # Load files
        # benigns = load_txt_file(os.path.join(name, 'benigns.txt'))
        # sybils = load_txt_file(os.path.join(name, 'sybils.txt'))
        nodes = load_txt_file(os.path.join(name, 'nodes.txt'))

        b_prime = load_txt_file(os.path.join(name, 'btrain.txt'))
        s_prime = load_txt_file(os.path.join(name, 'strain.txt'))
        edges = load_txt_file_for_edges(os.path.join(name, 'edges.txt'))

        resistances = load_json_file(os.path.join(name, 'resistanceDictionary.json'))
        probability_of_resistances = load_json_file(os.path.join(name, 'probability_of_resistance_dictionary.json'))

        start_time = time.time()
        
        degree_in_minus_S_B,in_neigs_minus_S_B,out_neigs = creat_in_degree_neigs_minus_S_B(edges,
                                                                                       nodes,
                                                                                       b_prime,
                                                                                       s_prime)
        

        if  'facebook' in name1:
            k = 40
        elif 'lastfm' in name1:
            k = 75
        elif 'Pokec' in name1:
            k = 100
        elif 'twitter' in name1:
            k = 100
        
        print("k is ",k)
        
        A = traversing_resistance_and_degree_aware(nodes = nodes,
            edges = edges,
            benigns = b_prime,
            pr = probability_of_resistances ,
            r = resistances,
            k = k,
            degree_in = degree_in_minus_S_B,
            in_neigs_minus_S_B = in_neigs_minus_S_B,
            out_neigs = out_neigs)

        ##
        #do discovering and new_B#
        newly_benigns = run_bfs_discovery(B=b_prime,
            A=A,
            r=resistances,
            in_neigs=in_neigs_minus_S_B)
        ##
        print("|newly_benigns|= ",len(newly_benigns))
        ##
        #write new_B
        with open(os.path.join(name, 'newly_benigns.txt'), 'w') as file:
            for item in newly_benigns:
                file.write(f"{item}\n")
        ##


        end_time = time.time()
        timer(start_time,end_time)



        ####################PAE####################
        start_time = time.time()

        benigns = b_prime[:] + newly_benigns[:]
        sybils = s_prime[:]

        

    
        degree_in, in_neigs = create_in_degree_neigs(edges, nodes, benigns, sybils)
        
        nodevalues = {}
        for node in benigns:
            nodevalues[node] = (1 - probability_of_resistances[node]) * degree_in[node]

        if  'facebook' in name1:
            k = 40
        elif 'lastfm' in name1:
            k = 75
        elif 'Pokec' in name1:
            k = 100
        elif 'twitter' in name1:
            k = 100

        print("k is ",k)

        A, value = top_k_nodes_sum(nodevalues, k)


        A_in_edges = []
        for node in A:
            if resistances[node] == 0: 
                
                for neig in in_neigs[node]:
                    A_in_edges.append([neig, node])

        print("|PAE|=",len(A_in_edges))
        


        end_time = time.time()
        timer(start_time,end_time)

        
        #Write PAE
        filename = os.path.join(name, 'PAE.txt')
        with open(filename, 'w') as file:
            for edge in A_in_edges:
                file.write(f"{edge[0]} {edge[1]}\n")







        