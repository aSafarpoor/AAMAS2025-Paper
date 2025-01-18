import heapq
import numpy as np
from tqdm import tqdm
import random
import math
import matplotlib.pyplot as plt
import json
import os

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
        return {int(float(k)): int(float(v)) for k, v in data.items()}



# Loop through datasets
for name1 in ['twitter+','facebook+', 'lastfm+','Pokec+']:
  for name3 in ['ba','bfs','random']:
    name = f'./datasets/{name1}{name3}/'
    benigns = load_txt_file(os.path.join(name, 'benigns.txt'))
    sybils = load_txt_file(os.path.join(name, 'sybils.txt'))

    if name1 == 'facebook+':
        lenstrain = 80#int(len(sybils)*0.5)
        lenbtrain = 80#int(len(benigns)*0.5)
    elif name1 == 'lastfm+':
        lenstrain = 150#int(len(sybils)*0.5)
        lenbtrain = 150#int(len(benigns)*0.5)
    elif name1 == 'Pokec+' :
        lenstrain = 200#int(len(sybils)*0.5)
        lenbtrain = 200#int(len(benigns)*0.5)
    elif name1 == 'twitter+':
        lenstrain = 200
        lenbtrain = 200
    lenstest = len(sybils) - lenstrain
    lenbtest = lenstest#len(benigns) - lenbtrain

    print(f"Length of btrain: {lenbtrain}, Length of strain: {lenstrain}, Length of btest: {lenbtest}, Length of stest: {lenstest}")

    random.seed(1)
    bsamples = benigns[:]#random.sample(benigns, lenbtrain+lenbtest)
    ssamples = sybils[:]#random.sample(sybils, lenstrain+lenstest)

    random.shuffle(bsamples)
    random.shuffle(ssamples)

    btrain = bsamples[:lenbtrain]
    btest = bsamples[lenbtrain:lenbtrain+lenbtest]
    strain = ssamples[:lenstrain]
    stest = ssamples[lenstrain:lenstrain+lenstest]

    for name2 in ['random', 'ba', 'bfs']:
        name = f'./datasets/{name1}{name2}/'    

        with open(os.path.join(name, 'btrain.txt'), 'w') as file:
            for node in btrain:
                file.write(f"{node}\n")
        with open(os.path.join(name, 'btest.txt'), 'w') as file:
            for node in btest:
                file.write(f"{node}\n")
        with open(os.path.join(name, 'strain.txt'), 'w') as file:
            for node in strain:
                file.write(f"{node}\n")
        with open(os.path.join(name, 'stest.txt'), 'w') as file:
            for node in stest:
                file.write(f"{node}\n")