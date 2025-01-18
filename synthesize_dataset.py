import pandas as pd
import numpy as np
import networkx as nx

import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from collections import deque
import tarfile

is_directed_flag = True

"""# prephase"""

with open('edges.txt', 'r') as file:
    osn = []
    for line in tqdm(file):
        numbers = line.split()
        osn.append([int(numbers[0]), int(numbers[1])])

dataSets = {
    'osn':{
        'nodes':[],
        'edges':[]
    }
}

def node_and_edge_dreator(ds,name,dataSets):

    edges = np.array(ds)
    nodes = list(set(edges.reshape(-1).tolist()))

    nodes.sort()
    idlist = {}
    indexer = 0
    for i in tqdm(range(len(nodes))):
        idlist[nodes[i]] = indexer
        nodes[i] = indexer
        indexer += 1

    for i in tqdm(range(len(edges))):
        edges[i][0] = idlist[edges[i][0]]
        if not is_directed_flag:
            edges[i][1] = idlist[edges[i][1]]


    print('\n',max(nodes),min(nodes),len(nodes),'\n\n')


    dataSets[name]['nodes'] = nodes[:]
    dataSets[name]['edges'] = edges[:]


node_and_edge_dreator(ds=osn,name="osn",dataSets=dataSets)

"""# Creat Benign Region and Sybil Region"""

def BenignRegionAndSybilRegion(nodes,edges,sybilToBenignRatio,rstar):

    numberOfNodes = len(nodes)

    G = nx.Graph()
    G.add_edges_from(edges)

    directed_G = nx.DiGraph()
    directed_G.add_edges_from(edges)

    # Get the list of edges
    edges = np.array(edges)
    nodes = np.array(nodes)
    lenbenigns = len(nodes)
    benignnodes = nodes[:]
    lensybils = int(lenbenigns*sybilToBenignRatio)
    nodes.sort()

    #creat sybil region
    sybilnodes = np.array([])
    maxnodes = int(max(nodes))

    # here we creat just sybil nodes, no edge no dual assignment
    for i in tqdm(range(1,1+lensybils)):
        nodes = np.append(nodes, maxnodes + i)
        sybilnodes = np.append(sybilnodes, maxnodes + i)

    # generate resistance
    resistance_dictionary = {}
    num_samples = len(benignnodes)
    vulnerable_benign_nodes = []
    for i in range(len(benignnodes)):
        resistance_dictionary[benignnodes[i]] = 1 if random.random() <= rstar else 0
        if resistance_dictionary[benignnodes[i]] == 0 :
            vulnerable_benign_nodes.append(benignnodes[i])
    # vulnerable_benign_nodes = []
    # for node in resistance_dictionary.keys():
    #     if resistance_dictionary[node] == 0:
    #         vulnerable_benign_nodes.append(node)

    #here we wants to assign list of sybils to benigns
    #later we can use this dual sets to run strategies and copy the current edges
    def dual_selection(sybils,undirectedG):
        listofnodes = undirectedG.nodes()
        visited = set()
        benignPrime = []
        queue = []
        while len(visited) < len(sybils) :
            if len(queue) == 0:
                flagTemp = True
                while flagTemp:
                    randomElement = np.array(random.sample(listofnodes, 1))[0]
                    if randomElement not in visited:
                        flagTemp = False
                        queue = [randomElement]

            currentNode = queue.pop(0)

            if currentNode not in visited:
                sample = int(currentNode)
                visited.add(currentNode)
                queue.extend(undirectedG.neighbors(sample))
        dual_pairs = np.column_stack((sybils, list(visited)))
        return dual_pairs

    dual_pairs = dual_selection(sybilnodes,undirectedG=G)

    dual_pairs_dictionary_b_to_s = {}
    dual_pairs_dictionary_s_to_b = {}

    for pair in tqdm(dual_pairs):
        s,b = pair[0],pair[1]
        dual_pairs_dictionary_b_to_s[b] = s
        dual_pairs_dictionary_s_to_b[s] = b

    sybiledges = []
    tempset = set(dual_pairs[:,1]) # tempset is the dual of sybils in benign

    rond_s =  0 # Number of edges from B' to B/B'

    for e in tqdm(edges):
        if e[0] in tempset:
            if e[1] in tempset:
                sybiledges.append([dual_pairs_dictionary_b_to_s[e[0]],
                                    dual_pairs_dictionary_b_to_s[e[1]]])
            else:
                rond_s +=1

    edges = np.concatenate((edges, sybiledges))

    return (edges,
            nodes,
            benignnodes,
            sybilnodes,
            rond_s,
            dual_pairs,
            resistance_dictionary,
            dual_pairs_dictionary_s_to_b,
            dual_pairs_dictionary_b_to_s,
            vulnerable_benign_nodes)

"""### .

here we have to keep these 2 variables:
* nodes
* edges

nodes will stay the same but allEdges which contains attacks will change later.
"""

def out_degree_minus_dual_of_s(nodes,edges,bdualofs):

    out_degree = {}
    for v in nodes:
        out_degree[v] = 0

    bdualofsset = set(bdualofs)
    passcounter = notpasscounter = 0
    for edge in tqdm(edges):
        if edge[1] in bdualofsset:
            passcounter+=1
        else:
            out_degree[edge[0]] += 1
            notpasscounter += 1

    # print(passcounter,notpasscounter)
    return out_degree

"""# First approach: Random Attacks"""

def FirstApproachRandomAttacksFirstScenario(edges,
                                            sybilNodes,
                                            benignNodes,
                                            rondS,
                                            resistanceDictionary,
                                            dualPairsDictionarySToB,
                                            c,
                                            bdualofs):
    allEdges = np.copy(edges)
    allEdges = allEdges.tolist()
    edgesAttack = np.array([[], []])
    # reverseEdgeCounter = 0

    khoob = bad = 0

    benignNodes.sort()
    setOfBenignNodes = set(benignNodes)
    benignEdges = [row for row in edges if row[0] in setOfBenignNodes and row[1] in setOfBenignNodes]
    # Just benign region := G1
    G1 = nx.DiGraph(benignEdges)

    nodes = list(set(np.array(edges).reshape(-1)))
    outdegreeminusdualofs = out_degree_minus_dual_of_s(nodes,edges,bdualofs)
    # print(outdegreeminusdualofs)

    #attack phase
    for u in tqdm(sybilNodes):
        # outDegreeOfNode = G1.out_degree(dualPairsDictionarySToB[u])

        checked = []
        # numberOfAcceptedNodes = 0
        shuffled_samples = np.random.choice(vulnerable_benign_nodes, size=len(vulnerable_benign_nodes), replace=False)
        indexofshaffellist = 0
        # while numberOfAcceptedNodes < outDegreeOfNode:
        # print(outdegreeminusdualofs[dualPairsDictionarySToB[u]])
        for _ in range(outdegreeminusdualofs[dualPairsDictionarySToB[u]]*c):
            try:
                sample = shuffled_samples[indexofshaffellist]
            except:
                break

            indexofshaffellist+=1
            

            # numberOfAcceptedNodes += 1
            if resistanceDictionary[sample]==0:
                newEdge = [u, sample]
                allEdges.append(newEdge)
                edgesAttack = np.append(edgesAttack, newEdge)

                if random.random() < 0.5:  # for reverse of attacks with p=0.5
                    # reverseEdgeCounter+=1
                    newEdge = [sample, u]
                    allEdges.append(newEdge)


    edgesAttack = np.array(edgesAttack).reshape([-1, 2])

    return edgesAttack, allEdges

"""# Second approach: modified BA"""

def SecondApproachBAFirstScenario(edges,
                                  sybilNodes,
                                  benignNodes,
                                  rondS,
                                  resistanceDictionary,
                                  dualPairsDictionarySToB,
                                  c):
    # help function
    def computeNodeProbabilitiesBasedOnBAIndegree(G):
        inDegrees = dict(G.in_degree())
        totalInDegree = sum(inDegrees.values())

        nodeProbabilities = {}
        for node in G.nodes():
            inDegreeProb = inDegrees[node] / totalInDegree if totalInDegree > 0 else 0
            nodeProbabilities[node] = inDegreeProb
        return nodeProbabilities

    #compute benignEdges:
    benignNodes.sort()
    setOfBenignNodes = set(benignNodes)
    benignEdges = [row for row in edges if row[0] in setOfBenignNodes and row[1] in setOfBenignNodes]

    G1 = nx.DiGraph(benignEdges)

    nodeProbabilities = computeNodeProbabilitiesBasedOnBAIndegree(G1)

    p1 = {}
    p2_counter = {}
    sum_p2_counter =  len(benignNodes)
    for node in benignNodes:
        p1[node] = nodeProbabilities[node]
        p2_counter[node] = 1


    allEdges = np.copy(edges)
    allEdges = allEdges.tolist()
    edgesAttack = np.array([[],[]])

    nodes = list(set(np.array(edges).reshape(-1)))
    # outdegreeminusB = out_degree_minus_B(nodes,edges,B=benignNodes)
    outdegreeminusdualofs = out_degree_minus_dual_of_s(nodes,edges,bdualofs)

    for u in tqdm(sybilNodes):
        # outDegreeOfNode = G1.out_degree(dualPairsDictionarySToB[u])
        checked = []
        # numberOfAcceptedNodes = 0

        p = {}
        for node in benignNodes:
            try:
                p[node] = p1[node] + p2_counter[node]/sum_p2_counter
            except:
                print(node,sum_p2_counter)

        plist = np.array([p[node] for node in benignNodes])
        plist = plist/sum(plist)

        shuffled_samples = np.random.choice(benignNodes, size=len(benignNodes), replace=False, p=plist)
        indexofshaffellist = 0

        # while numberOfAcceptedNodes < outDegreeOfNode:
        for _ in range(outdegreeminusdualofs[dualPairsDictionarySToB[u]]*c):
            # sample = random.choices(vulnerable_benign_nodes, weights=plist)[0]
            try:
                sample = shuffled_samples[indexofshaffellist]
            except:
                break
            indexofshaffellist += 1
            '''
            if sample in checked:
                    continue
            checked.append(sample)
            '''

            '''
            resistance = resistanceDictionary[sample]
            if resistance == 0:
            '''
            # numberOfAcceptedNodes += 1
            if  resistanceDictionary[sample]==0:
                newEdge = [u, sample]
                allEdges.append(newEdge)
                edgesAttack = np.append(edgesAttack, newEdge)
                sum_p2_counter +=1
                p2_counter[sample] += 1
                if random.random() < 0.5: # for reverse of attacks with p=0.5
                    # reverseEdgeCounter += 1
                    newEdge = [sample, u]
                    allEdges.append(newEdge)



    # edgesAttack = edgesAttack.reshape([-1, 2])
    edgesAttack = np.array(edgesAttack).reshape([-1, 2])

    return edgesAttack, allEdges

"""# Third approach: BFS"""

def ThirdApproachBFSFisrtScenario(edges,
                                  sybilNodes,
                                  benignNodes,
                                  rondS,
                                  resistanceDictionary,
                                  dualPairsDictionarySToB,
                                  c):

    bfscounter = bacounter = 0
    # help function
    def computeNodeProbabilitiesBasedOnBAIndegree(G):
        inDegrees = dict(G.in_degree())
        totalInDegree = sum(inDegrees.values())

        nodeProbabilities = {}
        for node in G.nodes():
            inDegreeProb = inDegrees[node] / totalInDegree if totalInDegree > 0 else 0
            nodeProbabilities[node] = inDegreeProb
        return nodeProbabilities

    allEdges = np.copy(edges)
    nodes = list(set(allEdges.reshape(-1)))
    allEdges = allEdges.tolist()

    # compute benignEdges:
    benignNodes.sort()
    setOfBenignNodes = set(benignNodes)
    benignEdges = [row for row in edges if row[0] in setOfBenignNodes and row[1] in setOfBenignNodes]
    # Just benign region := G1
    G1 = nx.DiGraph(benignEdges)

    #attack phase
    edgesAttack = []
    # outdegreeminusB = out_degree_minus_B(nodes,edges,B=benignNodes)
    outdegreeminusdualofs = out_degree_minus_dual_of_s(nodes,edges,bdualofs)

    for u in tqdm(sybilNodes):

        # outDegreeOfNode = G1.out_degree(dualPairsDictionarySToB[u])
        numberOfAcceptedNodes = 0

        visited = set()  # for BFS
        queue = [dualPairsDictionarySToB[u]]  # for BFS

        doBAFlag = False
        k = 0




        for _ in range(outdegreeminusdualofs[dualPairsDictionarySToB[u]]*c):
        # while numberOfAcceptedNodes < outDegreeOfNode and (not doBAFlag) :

            if doBAFlag:
                break
            # Do BFS
            try:
                currentNode = queue.pop(0)
            except:
                doBAFlag = True
                break


            if currentNode not in visited:
                sample = int(currentNode)
                visited.add(currentNode)
                queue.extend(G1.neighbors(sample))

                resistance = resistanceDictionary[sample]
                if  resistance == 0:
                    numberOfAcceptedNodes += 1
                    newEdge = [u, sample]
                    allEdges.append(newEdge)
                    edgesAttack = np.append(edgesAttack, newEdge)
                    bfscounter+=1
                    if random.random() < 0.5:  # for reverse of attacks with p=0.5
                        # reverseEdgeCounter += 1
                        newEdge = [sample, u]
                        allEdges.append(newEdge)

            k+=1

        if doBAFlag and numberOfAcceptedNodes < outdegreeminusdualofs[dualPairsDictionarySToB[u]]*c:

            temp = computeNodeProbabilitiesBasedOnBAIndegree(G1)
            p1 = []
            for node in benignNodes:
                p1.append(temp[node])
            p1 = np.array(p1)
            acceptedCounter = len(p1)
            p2 = np.ones(len(p1)) * (1/acceptedCounter)
            degreeTarget = {}
            for node in benignNodes:
                degreeTarget[node] = 1

            # while numberOfAcceptedNodes < outDegreeOfNode and len(visited)<len(benignNodes):
            for _ in range(numberOfAcceptedNodes,outdegreeminusdualofs[dualPairsDictionarySToB[u]]*c):
                p = (p1+p2)/2
                sample = random.choices(benignNodes, weights=p)[0]

                if sample in visited:
                    continue
                visited.add(sample)
                resistance = resistanceDictionary[sample]

                if resistance == 0:
                    numberOfAcceptedNodes += 1
                    bacounter+=1
                    newEdge = [u, sample]
                    allEdges.append(newEdge)
                    edgesAttack = np.append(edgesAttack, newEdge)
                    degreeTarget[sample] += 1
                    if random.random() < 0.5: # for reverse of attacks with p=0.5
                        # reverseEdgeCounter += 1
                        newEdge = [sample, u]
                        allEdges.append(newEdge)


    # edgesAttack = edgesAttack.reshape([-1, 2])
    edgesAttack = np.array(edgesAttack).reshape([-1, 2])

    print('\n',bfscounter,bacounter)
    return edgesAttack, allEdges

"""#Run :)

##save general
"""

import json

def save_common_data(nodes,
                     sybil_nodes,
                     benign_nodes,
                     dualPairsDictionaryBToS,
                     strategyName,
                     sourcedatasetName ,
                     resistanceDictionary):

    base_path = "dataset/"

    # Save nodes
    nodes_filename = f"{base_path}nodes.txt"
    np.savetxt(nodes_filename, nodes, fmt='%d')

    # Save sybil nodes
    sybil_nodes_filename = f"{base_path}sybils.txt"
    np.savetxt(sybil_nodes_filename, sybil_nodes, fmt='%d')

    # Save benign nodes
    benign_nodes_filename = f"{base_path}benigns.txt"
    np.savetxt(benign_nodes_filename, benign_nodes, fmt='%d')

    # Save dualPairsDictionaryBToS
    dualPairsDict_filename = f"{base_path}dualPairsDictionaryBToS.json"
    with open(dualPairsDict_filename, 'w') as file:
        json.dump(dualPairsDictionaryBToS, file, indent=4)

    # Save resistanceDictionary
    temp = {int(k) if isinstance(k, np.int64) else k: v for k, v in resistanceDictionary.items()}
    temp = {int(k) if isinstance(k, np.int32) else k: v for k, v in resistanceDictionary.items()}

    resistanceDict_filename = f"{base_path}resistanceDictionary.json"
    with open(resistanceDict_filename, 'w') as file:
        json.dump(temp, file, indent=4)


    return f"Files saved in {base_path} with strategy name '{strategyName}' and source dataset name '{sourcedatasetName}'"

def save_data_edges(edges, strategyName, sourcedatasetName):
    # Save edges
    base_path = "dataset/"

    edges_filename = f"{base_path}edges_{strategyName}_{sourcedatasetName}.txt"
    np.savetxt(edges_filename, edges, fmt='%d')

    return "down"

"""#### next"""

import os

os.makedirs("dataset", exist_ok=True)

rstar = 0.75
sybilToBenignRatio = 0.10
c = 4

def runMoreGeneral(SOURCE_DATASET_NAME,dataSets):
    # Initialize data
    edges = dataSets[SOURCE_DATASET_NAME]['edges']
    nodes = dataSets[SOURCE_DATASET_NAME]['nodes']
    average_out_degree = len(edges) / len(nodes)
    print("Average Out Degree is:", average_out_degree)

    # Process regions
    x = BenignRegionAndSybilRegion(
        nodes=nodes,
        edges=edges,
        sybilToBenignRatio = sybilToBenignRatio,
        rstar=rstar
    )
    edges = x[0]
    nodes = x[1]
    benign_nodes = x[2]
    sybil_nodes = x[3]
    rond_s = x[4]
    dual_pairs = x[5]
    resistance_dictionary = x[6]
    dual_pairs_dict_s_to_b = x[7]
    dual_pairs_dict_b_to_s = x[8]
    vulnerable_benign_nodes = x[9]


    # Save common data
    result = save_common_data(nodes,
                            sybil_nodes,
                            benign_nodes,
                            dual_pairs_dict_b_to_s,
                            "",
                            SOURCE_DATASET_NAME,
                            resistance_dictionary)
    print(result)
    return (nodes,
            edges,
            rond_s,
            sybil_nodes,
            benign_nodes,
            dual_pairs_dict_b_to_s,
            dual_pairs_dict_s_to_b,
            resistance_dictionary,
            vulnerable_benign_nodes)

"""## save special"""

import zipfile
import os

def zip_all_files_in_folder(folder_path, output_zip_file,name):
    with zipfile.ZipFile(output_zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if name in file:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, arcname=os.path.relpath(file_path, folder_path))

    print(f"Created zip file '{output_zip_file}' containing all files from '{folder_path}'.")

"""## temp test"""

for SOURCE_DATASET_NAME in ["osn"]:

    (nodes,
    edges,
    rond_s,
    sybil_nodes,
    benign_nodes,
    dual_pairs_dict_b_to_s,
    dual_pairs_dict_s_to_b,
    resistance_dictionary,
     vulnerable_benign_nodes) = runMoreGeneral(SOURCE_DATASET_NAME,dataSets)

    bdualofs = list(map(int,list(dual_pairs_dict_b_to_s.keys())))


    # Random attack approach

    all_edges = np.copy(edges)


    edges_attack, all_edges = FirstApproachRandomAttacksFirstScenario(
        all_edges,
        sybil_nodes,
        benign_nodes,
        rond_s,
        resistance_dictionary,
        dual_pairs_dict_s_to_b,
        c,
        bdualofs)

    # pos = Drawer(
    #    allEdges = np.copy(all_edges),
    #    sybilNodes = np.copy(sybil_nodes),
    #    benignNodes = np.copy(benign_nodes),
    #    dualPairsDictionarySToB = dual_pairs_dict_s_to_b,
    #    titleofdraw = "random facebook",
    #    pos = '-1')

    print("Random attack results:", save_data_edges(all_edges, "Random", SOURCE_DATASET_NAME))

    # BA approach

    all_edges = np.copy(edges)
    edges_attack, all_edges = SecondApproachBAFirstScenario(
        all_edges,
        sybil_nodes,
        benign_nodes,
        rond_s,
        resistance_dictionary,
        dual_pairs_dict_s_to_b,
        c)

    # pos = Drawer(
    #    allEdges = np.copy(all_edges),
    #    sybilNodes = np.copy(sybil_nodes),
    #    benignNodes = np.copy(benign_nodes),
    #    dualPairsDictionarySToB = dual_pairs_dict_s_to_b,
    #    titleofdraw = "BA facebook",
    #    pos = pos)

    print("BA approach results:", save_data_edges(all_edges, "BA", SOURCE_DATASET_NAME))
    print(len(edges), len(all_edges))

    # BFS approach

    all_edges = np.copy(edges)
    edges_attack, all_edges = ThirdApproachBFSFisrtScenario(
        all_edges,
        sybil_nodes,
        benign_nodes,
        rond_s,
        resistance_dictionary,
        dual_pairs_dict_s_to_b,
        c)

    # pos = Drawer(
    #    allEdges = np.copy(all_edges),
    #    sybilNodes = np.copy(sybil_nodes),
    #    benignNodes = np.copy(benign_nodes),
    #    dualPairsDictionarySToB = dual_pairs_dict_s_to_b,
    #    titleofdraw = "BFS facebook",
    #    pos = pos)

    print("BFS approach results:", save_data_edges(all_edges, "BFS", SOURCE_DATASET_NAME))
    print(len(edges), len(all_edges))




    folder_path = 'dataset'
    output_zip_file = 'synthesizedDataset.zip'
    zip_all_files_in_folder(folder_path, output_zip_file, SOURCE_DATASET_NAME)

