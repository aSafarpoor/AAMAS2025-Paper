

import random
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
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
        return {int(float(k)): int(float(v)) for k, v in data.items()}



class Data:
    def __init__(self,network_content,
                 pae_network_content,
                 train_set_content,
                 test_set_content,
                 post_file,
                 prior_content):

        self.network_content = network_content
        self.pae_network_content=set(tuple(inner_list) for inner_list in pae_network_content)
        self.train_set_content = train_set_content
        self.test_set_content = test_set_content
        self.prior_content = prior_content
        self.post_file = post_file
        self.theta_pos = 0.9
        self.theta_neg = 0.1
        self.theta_unl = 0.5
        self.weighted_graph = 0
        self.weight = 0.9
        self.weight2 = 0.5

        self.max_iter = 2

        self.network_map = defaultdict(list)
        self.post = None
        self.post_pre = None
        self.prior = None
        self.ordering_array = None
        self.N = 0

    def add_edge(self, node1, node2, w):
        if node1 == node2:
            return
        self.network_map[node1].append((node2, w))

    def read_network(self):
        maxnode = -1
        if len(self.pae_network_content)>0:
            for line in self.network_content:
                # parts = line.split()
                node1 = int(line[0])
                node2 = int(line[1])
                if node1>maxnode:
                    maxnode = node1
                if node2>maxnode:
                    maxnode = node2
                w = self.weight - 0.5
                if (node1,node2) in self.pae_network_content:
                    w = self.weight2 - 0.5
                    
                self.add_edge(node1, node2, w)
        else:
            for line in self.network_content:
                # parts = line.split()
                node1 = int(line[0])
                node2 = int(line[1])
                if node1>maxnode:
                    maxnode = node1
                if node2>maxnode:
                    maxnode = node2
                w = self.weight - 0.5
                self.add_edge(node1, node2, w)


        self.N = max([len(self.network_map) , maxnode])+1
        self.post = np.zeros(self.N)
        self.post_pre = np.zeros(self.N)
        self.prior = np.zeros(self.N)


    
    def read_prior(self):
        self.prior.fill(self.theta_unl - 0.5)

        if self.prior_content:
            for line in self.prior_content:
                parts = line.split()
                node = int(parts[0])
                score = float(parts[1])
                self.prior[node] = score - 0.5

        if self.train_set_content:
            lines = self.train_set_content
            pos_train_str = lines[0]
            neg_train_str = lines[1]

            for sub in pos_train_str:
                self.prior[int(sub)] = self.theta_pos - 0.5

            for sub in neg_train_str:
                self.prior[int(sub)] = self.theta_neg - 0.5

    def write_posterior(self):
        with open(self.post_file, 'w') as f:
            for i in range(self.N):
                f.write(f"{i} {self.post[i] + 0.5:.10f}\n")

    def test(self,extrabenign=[]):
        def acc(true_benign, true_sybil, false_benign, false_sybil, unknown_sybil, unknown_benign):
            return (true_sybil + true_benign) / (true_sybil + true_benign + false_sybil + false_benign + unknown_sybil + unknown_benign)

        lines = self.test_set_content
        
        sybils = list(map(int, lines[0]))
        benigns = list(map(int, lines[1]))
        
        # benigns = list(set(benigns).union(extrabenign))
        # sybils = list(set(sybils) - set(extrabenign))
        

        
        ts, fs, tb, fb, us, ub = 0, 0, 0, 0, 0, 0
        labels = []
        predictions = []

        for node in extrabenign:
            self.post[node] = -1

        for node in sybils:
            labels.append(1)
            predictions.append(1 if self.post[node] > 0 else 0)
            if self.post[node] > 0:
                ts += 1
            elif self.post[node] < 0:
                fs += 1
            else:
                us += 1

        for node in benigns:
            labels.append(0)
            predictions.append(1 if self.post[node] > 0 else 0)
            if self.post[node] > 0:
                fb += 1
            elif self.post[node] < 0:
                tb += 1
            else:
                ub += 1

        # print(f"True benigns: {tb}, True sybils: {ts}")
        # print(f"False benigns: {fb}, False sybils: {fs}")
        # print(f"Unknown sybils: {us}, Unknown benigns: {ub}")

        # accuracy = acc(tb, ts, fb, fs, us, ub)
        # print("Accuracy: ", round(accuracy,3))

        # Calculate AUC
        try:
            auc = roc_auc_score(labels, predictions)
        except:
            auc = 0.50
        print("AUC: ", round(auc,3))

        # Calculate precision, recall, and F1 score
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        
        '''
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 Score: ", f1)'''

    def lbp(self):
        self.ordering_array = list(range(self.N))
        np.copyto(self.post, self.prior)

        for iter in range(self.max_iter):
            np.copyto(self.post_pre, self.post)
            random.shuffle(self.ordering_array)

            for index in range(self.N):
                node = self.ordering_array[index]
                self.post[node] = sum(2 * self.post_pre[nei] * w 
                    for nei, w in self.network_map[node])
                self.post[node] += self.prior[node]
                self.post[node] = min(0.5, max(-0.5, self.post[node]))

random.seed()

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

for name1 in ['twitter+','Pokec+','facebook+', 'lastfm+']:
    for name2 in ['random', 'ba', 'bfs']:
    # for name2 in ['ba']:

        name = f'./datasets/{name1}{name2}/'
        print("\n\n", name1,name2)
        network_content = load_txt_file_for_edges(name + 'edges.txt')
        pae_network_content = load_txt_file_for_edges(name + 'PAE.txt')

        btrain = load_txt_file(name + 'btrain.txt')
        btrain2 = load_txt_file(name + 'newly_benigns.txt')
        strain = load_txt_file(name + 'strain.txt')
        train_set_content = [strain,btrain]

        btest = load_txt_file(name + 'btest.txt')
        stest = load_txt_file(name + 'stest.txt')
        test_set_content = [stest,btest]

        

        prior_content = read_file(name + 'prior.txt') if 'prior.txt' in os.listdir(name) else []


        start_time = time.time()
        print("init")
        data = Data(network_content=network_content,
                    pae_network_content=[],
                    train_set_content=train_set_content,
                    test_set_content=test_set_content,
                    post_file='post_SybilSCAR.txt',
                    prior_content=prior_content)
        data.read_network()
        data.read_prior()
        data.lbp()
        end_time = time.time()
        # timer(start_time,end_time)

        start_time = time.time()
        print("init")
        data = Data(network_content=network_content,
                    pae_network_content=[],
                    train_set_content=train_set_content,
                    test_set_content=test_set_content,
                    post_file='post_SybilSCAR.txt',
                    prior_content=prior_content)
        data.read_network()
        data.read_prior()
        data.lbp()
        # data.write_posterior()
        data.test()
        end_time = time.time()
        timer(start_time,end_time)


        start_time = time.time()
        print("MB")
        # for node in btrain2:
            # btrain.append(node)
        train_set_content = [strain,btrain]
        data = Data(network_content=network_content,
                    pae_network_content=[],
                    train_set_content=train_set_content,
                    test_set_content=test_set_content,
                    post_file='post_SybilSCAR.txt',
                    prior_content=prior_content)
        data.read_network()
        data.read_prior()
        data.lbp()
        # data.write_posterior()
        data.test(extrabenign = btrain2)
        end_time = time.time()
        timer(start_time,end_time)
        

        start_time = time.time()
        print("MB+PAE")
        train_set_content = [strain,btrain]
        data = Data(network_content=network_content,
                    pae_network_content=pae_network_content,
                    train_set_content=train_set_content,
                    test_set_content=test_set_content,
                    post_file='post_SybilSCAR.txt',
                    prior_content=prior_content)
        data.read_network()
        data.read_prior()
        data.lbp()
        # data.write_posterior()
        data.test(extrabenign = btrain2)
        end_time = time.time()
        timer(start_time,end_time)
        

        