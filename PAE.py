import heapq
import numpy as np
from tqdm import tqdm
import random
import math
import matplotlib.pyplot as plt
import os
import json

# Function to load text file into a list
def load_txt_file(filename):
    with open(filename, 'r') as file:
        return [int(line.strip()) for line in file]

def load_txt_file_for_edges(filename):
    with open(filename, 'r') as file:
        return [list(map(int,line.strip().split())) for line in file]

# Function to load JSON file into a dictionary
def load_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        return {int(float((k))): int(float((v))) for k, v in data.items()}


def creat_in_degree_neigs(edges,nodes,b_prime,s_prime):
    degree_in = {}
    in_neigs = {}
    for node in nodes:
        degree_in[node] = 0
        in_neigs[node] = []

    setbands = set(b_prime).union(set(s_prime))

    for edge in tqdm(edges):
        if edge[0] not in setbands:
            try:
                degree_in[edge[1]] += 1
                in_neigs[edge[1]].append(edge[0])
            except:
                print("edgeeeee ",edge)

    return degree_in,in_neigs

def top_k_nodes_sum(d, k):
    top_k = heapq.nlargest(k, d.items(), key=lambda x: x[1])
    A = [key for key, value in top_k]
    top_k_sum = sum(value for key, value in top_k)
    return A,top_k_sum


def main(edges,nodes,b_prime,s_prime,benigns,sybils,s_zegond):

    degree_in,in_neigs = creat_in_degree_neigs(edges,nodes,b_prime,s_prime)

    nodevalues = {}
    nodesybiler = {}
    ratio_of_nodes = {}

    for node in b_prime:
        value = (1-probability_of_resistance[node]) * (degree_in[node])
        nodevalues[node] = value

        s = 0
        for x in in_neigs[node]:
            if x in s_zegond:
                s+=1
        nodesybiler[node] = s
        ratio_of_nodes[node] = s*100/max(1,degree_in[node])



    ks = list(range(1,len(b_prime),2))
    ks.append(len(b_prime))
    greedyoutput = []
    randomoutput = []
    optoutput = []
    greedysybils = []
    randomsybils = []
    greedyratio = []
    randomratio = []
    optratio = []

    for k in ks:

        if k > len(b_prime):
            break

        A,A_sum = top_k_nodes_sum(nodevalues, k)
        asum = 0
        bsum = 0
        for node in A:
            asum += nodesybiler[node]
            bsum += degree_in[node]
        greedysybils.append(asum)
        greedyoutput.append(bsum)
        greedyratio.append(round(asum*100/max(1,bsum),2))


        randomsamples = random.sample(b_prime, k)
        asum = 0
        bsum = 0
        for node in randomsamples:
            asum += nodesybiler[node]
            bsum += degree_in[node]
        randomsybils.append(asum)
        randomoutput.append(bsum)
        randomratio.append(round(asum*100/max(1,bsum),2))

        opt_nodes = heapq.nlargest(k, ratio_of_nodes, key=ratio_of_nodes.get)
        asum = 0
        bsum = 0
        for node in opt_nodes:
            asum += nodesybiler[node]
            bsum += degree_in[node]
        optoutput.append(bsum)
        optratio.append(round(asum*100/max(1,bsum),2))

    return greedyoutput,randomoutput,optoutput,greedyratio,randomratio,optratio,ks,randomsybils,greedysybils,ks

data = {}
for name1 in ['twitter+','facebook+', 'lastfm+','Pokec+']:
    greedyoutput_dictionary = {}
    randomoutput_dictionary = {}
    optoutput_dictionary = {}

    greedyratio_dictionary = {}
    randomratio_dictionary = {}
    optratio_dictionary = {}
    gs_dictionary = {}
    rs_dictionary = {}
    ks_dictionary = {}

    for name3 in ['ba','bfs','random']:
        name = f'./datasets/{name1}{name3}/'

        benigns = load_txt_file(os.path.join(name, 'benigns.txt'))
        sybils = load_txt_file(os.path.join(name, 'sybils.txt'))
        nodes = load_txt_file(os.path.join(name, 'nodes.txt'))

        b_prime = load_txt_file(os.path.join(name, 'btrain.txt'))


        print(len(b_prime))

        s_prime = load_txt_file(os.path.join(name, 'strain.txt'))
        resistances = load_json_file(os.path.join(name, 'resistanceDictionary.json'))
        probability_of_resistance = load_json_file(os.path.join(name, 'probability_of_resistance_dictionary.json'))
        edges = load_txt_file_for_edges(os.path.join(name, 'edges.txt'))

        total_count = len(sybils)
        s_zegond = [item for item in sybils if item not in s_prime]

        greedyoutput,randomoutput,optoutput,greedyratio,randomratio,optratio,ks,rs,gs,ks = main(edges,nodes,b_prime,s_prime,benigns,sybils,s_zegond)

        name_of_strategy_method = name3
        greedyoutput_dictionary[name_of_strategy_method] = greedyoutput
        randomoutput_dictionary[name_of_strategy_method] = randomoutput
        optoutput_dictionary[name_of_strategy_method] = optoutput
        greedyratio_dictionary[name_of_strategy_method] = greedyratio
        randomratio_dictionary[name_of_strategy_method] = randomratio
        optratio_dictionary[name_of_strategy_method] = optratio
        gs_dictionary[name_of_strategy_method] = gs
        rs_dictionary[name_of_strategy_method] = rs
        ks_dictionary[name_of_strategy_method] = ks

    data[name1] = {
                'greedyoutput_dictionary': greedyoutput_dictionary.copy(),
                'randomoutput_dictionary': randomoutput_dictionary.copy(),
                'optoutput_dictionary': optoutput_dictionary.copy(),
                'greedyratio': greedyratio_dictionary.copy(),
                'randomratio': randomratio_dictionary.copy(),
                'optratio': optratio_dictionary.copy(),
                'ks': ks_dictionary.copy()}


# for name1 in ['twitter+','facebook+', 'lastfm+','Pokec+']:
#     print("\n\n",name1,"\n\n")

#     out = data[name1]
#     fig, axes = plt.subplots(2, 3, figsize=(10, 6))  # 2 rows, 3 columns

#     # Titles for each column
#     column_titles = ['Random', 'Preferential Attachment', 'BFS']

#     # Plot data
#     titles = ["Proposed", "Random", "Greedy"]

#     for row_index, row in enumerate(['greedyoutput_dictionary',
#                                     'randomoutput_dictionary',
#                                     'optoutput_dictionary']):

#         for col_index, col in enumerate(['random', 'ba', 'bfs']):
#             ks = out['ks'][col]

#             ax = axes[row_index // 3, col_index]
#             ax.plot(ks,out[row][col],label=titles[row_index])
#             ax.set_title(f"{column_titles[col_index]}")
#             ax.set_xlabel('Budget')
#             ax.set_ylabel('Number of PAEs')
#             ax.legend(loc='lower right')


#     for row_index, row in enumerate(['greedyratio', 'randomratio', 'optratio']):
#         for col_index, col in enumerate(['random', 'ba', 'bfs']):
#             ks = out['ks'][col]
#             ax = axes[row_index // 3 + 1, col_index]
#             ax.plot(ks,out[row][col],label=titles[row_index])
#             # ax.set_title(f"{column_titles[col_index]} - {row.split('_')[0]}")
#             ax.set_xlabel('Budget')
#             ax.set_ylabel('Percentage of Attack Edges')
#             ax.legend(loc='upper right')

#     plt.tight_layout()
#     plt.show()



import matplotlib.pyplot as plt

for name1 in ['twitter+', 'facebook+', 'lastfm+', 'Pokec+']:
    print("\n\n", name1, "\n\n")

    out = data[name1]
    fig, axes = plt.subplots(2, 3, figsize=(6, 5))  # Adjusted figsize for a better layout

    # Titles for each column
    column_titles = ['Random\nStrategy', 'Preferential Attachment\nStrategy', 'BFS\nStrategy']

    # Plot data
    titles = ["Proposed", "Random", "Greedy"]

    # Colors and font properties
    xlabel_color = 'darkblue'
    ylabel_color = 'darkblue'
    font_weight = 'medium'  # Changed font weight to 'medium'

    # List to collect all handles and labels for a single legend
    handles, labels = [], []

    # Plotting the first row of subplots
    for row_index, row in enumerate(['greedyoutput_dictionary',
                                     'randomoutput_dictionary',
                                     'optoutput_dictionary']):
        for col_index, col in enumerate(['random', 'ba', 'bfs']):
            ks = out['ks'][col]
            ax = axes[row_index // 3, col_index]
            line, = ax.plot(ks, out[row][col], label=titles[row_index])
            ax.set_title(f"{column_titles[col_index]}")

            # Setting bold and colorful labels
            ax.set_xlabel('Budget', fontweight=font_weight, color=xlabel_color)
            ax.set_ylabel('Number of PAEs', fontweight=font_weight, color=ylabel_color)

            # Collect handles and labels for the legend only once
            if col_index == 0:  # Avoid duplicates by collecting from the first column only
                handles.append(line)
                labels.append(titles[row_index])

    # Plotting the second row of subplots
    for row_index, row in enumerate(['greedyratio', 'randomratio', 'optratio']):
        for col_index, col in enumerate(['random', 'ba', 'bfs']):
            ks = out['ks'][col]
            ax = axes[row_index // 3 + 1, col_index]
            line, = ax.plot(ks, out[row][col], label=titles[row_index])

            # Setting bold and colorful labels
            ax.set_xlabel('Budget', fontweight=font_weight, color=xlabel_color)
            ax.set_ylabel('Percentage of Attack Edges', fontweight=font_weight, color=ylabel_color)

    # Create a single legend for the entire figure below all subplots
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, fontsize=12)

    # Adjust layout to make space for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Adjust bottom space to fit the legend below
    plt.show()

