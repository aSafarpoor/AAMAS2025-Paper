import heapq
import numpy as np
from tqdm import tqdm
import random
import math
import matplotlib.pyplot as plt

import json
import os

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
        return {int(float((k))): float((v)) for k, v in data.items()}


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


"""# (BFS) Run MB by A:"""



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





def random_selection(B,k):
    random_elements = random.sample(B, min(k,len(B)))
    return random_elements

def highest_resistance(B,pr,k):
    pr_of_B = {}
    for node in B:
        pr_of_B[node]= pr[node]
    top_k = heapq.nlargest(min(k,len(B)), pr_of_B, key=pr_of_B.get)
    return top_k

def highest_resistance_and_degree(B,pr,k,degree_in_minus_S_B):
    score_of_B = {}
    for node in B:
        score_of_B[node]= pr[node]*degree_in_minus_S_B[node]
    top_k = heapq.nlargest(min(k,len(B)), score_of_B, key=score_of_B.get)
    return top_k



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

"""# Choose **A** by **Monte Carlo**"""

def random_r_based_on_pr(x,pr):
    try:
        if random.random() > pr[x]:
            return 0
        return 1
    except:
        return 0

def estimate_function(B,pr,r,in_neigs,A,R,revealed={}):

    totalcounter = 0

    for _ in range(R):

        on_nodes = []
        on_B_nodes = []

        for node in A:
            if node in revealed.keys():
                if revealed[node] == 1:
                    on_nodes.append(node)
                if node in B:
                    on_B_nodes.append(node)

            elif random_r_based_on_pr(node,pr) == 1:
                on_nodes.append(node)
                if node in B:
                    on_B_nodes.append(node)

        newly_labeled = []
        visited = []


        for root in on_B_nodes:

            queue = [root]
            visited.append(root)

            while len(queue) > 0:
                current = queue.pop(0)

                neigs = in_neigs[current]
                for neig in neigs:
                    if neig in on_nodes and neig not in visited:
                        queue.append(neig)
                    visited.append(neig)

                    if neig not in B:
                        newly_labeled.append(neig)

        newly_labeled = list(set(newly_labeled))
        totalcounter += len(newly_labeled)


    return totalcounter/R


def choose_greedy(V,B,k,pr,r,in_neigs,R):
    A = []
    Vminus = V.copy()

    for _ in range(k):
        A.append(-1)
        maxestimation = [-1,-1]
        for v in tqdm(Vminus):
             A[-1] = v
             val = estimate_function(B,pr,r,in_neigs,A,R)
             if val > maxestimation[1]:
                maxestimation = [v,val]

        Vminus.remove(maxestimation[0])
        A[-1] = maxestimation[0]

    return A

def choose_greedy_reistance_aware(V,B,k,pr,r,in_neigs,R):
    A = []
    Vminus = V.copy()
    revealed = {}
    for _ in range(k):
        A.append(-1)
        maxestimation = [-1,-1]
        for v in tqdm(Vminus):
             A[-1] = v
             val = estimate_function(B,pr,r,in_neigs,A,R,revealed)
             if val > maxestimation[1]:
                maxestimation = [v,val]

        Vminus.remove(maxestimation[0])
        A[-1] = maxestimation[0]
        revealed[maxestimation[0]] = r[maxestimation[0]]

    return A

def main(in1,in2,path):

    benigns = load_txt_file(path+'benigns.txt')
    sybils = load_txt_file(path+'sybils.txt')
    nodes = load_txt_file(path+'nodes.txt')

    b_prime = load_txt_file(path+'btrain.txt')
    s_prime = load_txt_file(path+'strain.txt')
    resistances = load_json_file(path+'resistanceDictionary.json')
    probability_of_resistance = load_json_file(path+'probability_of_resistance_dictionary.json')
    edges = load_txt_file_for_edges(path+'edges.txt')

    degree_in_minus_S_B,in_neigs_minus_S_B,out_neigs = creat_in_degree_neigs_minus_S_B(edges,
                                                                                       nodes,
                                                                                       b_prime,
                                                                                       s_prime)

    print(len(b_prime),len(s_prime),len(benigns))

    outputs = {}

    for k in [1]+list(range(2,63,4)):
        As = {}

        
        As['MC-v1'] = choose_greedy(V = nodes,
                            B = b_prime,
                            pr = probability_of_resistance,
                            r = resistances,
                            k = k,
                            in_neigs = in_neigs_minus_S_B,
                            R = 100)
        
       

        As['MC-v2'] = choose_greedy_reistance_aware(V = nodes,
                            B = b_prime,
                            pr = probability_of_resistance,
                            r = resistances,
                            k = k,
                            in_neigs = in_neigs_minus_S_B,
                            R = 100)


        As['random'] = random_selection(B = b_prime,
                                         k = k)

        As['hr'] = highest_resistance(B = b_prime,
                                        pr = probability_of_resistance,
                                        k = k)

        As['hrd'] = highest_resistance_and_degree(B = b_prime,
                                                    pr = probability_of_resistance,
                                                    k = k,
                                                    degree_in_minus_S_B = degree_in_minus_S_B)


        As['tra'] = traversing_resistance_and_degree_aware(nodes,
                                                           edges,
                                                           b_prime.copy(),
                                                           probability_of_resistance,
                                                           resistances,
                                                           k,
                                                           degree_in_minus_S_B.copy(),
                                                           in_neigs_minus_S_B.copy(),
                                                           out_neigs.copy())


        print(As)

        outputs[k]={}
        for key in As.keys():
            out = run_bfs_discovery(B = b_prime,
                            A = As[key],
                            r = resistances,
                            in_neigs = in_neigs_minus_S_B)

            outputs[k][key] = out

    _data = outputs



    # Prepare data for plotting Facebook
    budgets_fb = list(_data.keys())
    mc_real_values_fb = [_data[budget]['MC-v1'] for budget in budgets_fb]
    mc_cheat_values_fb = [_data[budget]['MC-v2'] for budget in budgets_fb]
    random_values_fb = [_data[budget]['random'] for budget in budgets_fb]
    hr_values_fb = [_data[budget]['hr'] for budget in budgets_fb]
    hrd_values_fb = [_data[budget]['hrd'] for budget in budgets_fb]
    tra_values_fb = [_data[budget]['tra'] for budget in budgets_fb]


    # Plot Facebook data
    plt.figure(figsize=(5, 4),dpi=200)

    plt.plot(budgets_fb, mc_cheat_values_fb,label='Resistance Aware Monte Carlo Greedy')
    plt.plot(budgets_fb, mc_real_values_fb, label='Monte Carlo Greedy', color = 'black')
    plt.plot(budgets_fb, tra_values_fb,     label='Traversing', color= 'red')
    plt.plot(budgets_fb, hrd_values_fb,     label='Highest-Resistance-and-Degree')
    plt.plot(budgets_fb, hr_values_fb,      label='Highest-Resistance')
    plt.plot(budgets_fb, random_values_fb,  label='Random', color = 'olive')

    
    if in2 == "ba":
        in2 = "Preferential Attachment"
    plt.title(in2 + ' Attack Strategy') #Preferential Attachment
    plt.xlabel('Budget')
    plt.ylabel('Discovered Benigns')

    plt.legend(loc='upper left',framealpha=0.6)
    # plt.legend()


    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()


in1 = input('facebook,lastfm,pokec,twitter')
in2 = input('ba,bfs,random')
main(in1,in2,path=os.path.join(os.getcwd(), "datasets\\"+in1+"+"+in2+"\\"))