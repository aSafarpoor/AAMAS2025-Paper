
from tqdm import tqdm
import numpy as np
import networkx as nx
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import shuffle

import time
def timer(start_time,end_time, eshtrakat = 0):
    elapsed_time_s = end_time + eshtrakat - start_time
    elapsed_time_ms = elapsed_time_s * 1000
    print(f"Elapsed time in milliseconds: {elapsed_time_ms:.6f} ms")

def load_txt_file(filename):
    with open(filename, 'r') as file:
        return [int(line.strip()) for line in file]

def load_txt_file_for_edges(filename):
    with open(filename, 'r') as file:
        return [list(map(int,line.strip().split())) for line in file]


# def timed(func):
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
#         return result
#     return wrapper

class ModelUnweighted:

    def __init__(self, G, subset_nodes):
        """
        Initialize the ModelUnweighted class with a graph and a subset of nodes.

        :param G: The input graph (a networkx graph)
        :param subset_nodes: A list or set of nodes to compute metrics for
        """
        self.G = G
        self.subset_nodes = subset_nodes

    #@timed
    def compute_in_degrees(self):
        return {node: self.G.in_degree(node) for node in self.subset_nodes}

    #@timed
    def compute_out_degrees(self):
        return {node: self.G.out_degree(node) for node in self.subset_nodes}

    #@timed
    def compute_betweenness(self):
        betweenness = nx.betweenness_centrality_subset(self.G, sources=self.subset_nodes, targets=self.subset_nodes)
        return {node: betweenness[node] for node in self.subset_nodes}

    def compute_harmonic(self):
        harmonic = {}
        for node in tqdm(self.subset_nodes):
            lengths = nx.single_source_shortest_path_length(self.G, node)
            harmonic[node] = sum(1 / length for target, length in lengths.items() if target != node)
        return harmonic

    #@timed
    def compute_eigenvector(self):
        eigenvector = nx.eigenvector_centrality(self.G)
        return {node: eigenvector[node] for node in self.subset_nodes}

    #@timed
    def compute_katz(self, alpha=0.1, beta=1.0, tol=1e-6, max_iter=1000):
        katz = nx.katz_centrality_numpy(self.G, alpha=alpha, beta=beta)

        min_bound = -1e6
        max_bound = 1e6
        katz = {node: max(min(katz[node], max_bound), min_bound) for node in katz}

        return {node: katz[node] for node in self.subset_nodes}

    #@timed
    def compute_pagerank(self):
        pagerank = nx.pagerank(self.G)
        return {node: pagerank[node] for node in self.subset_nodes}

    #@timed
    def compute_clustering(self):
        return {node: nx.clustering(self.G, node) for node in self.subset_nodes}

    #@timed
    def compute_avg_shortest_path_length(self):
        avg_lengths = {}
        for node in tqdm(self.subset_nodes):
            lengths = nx.single_source_shortest_path_length(self.G, node)
            avg_lengths[node] = sum(lengths.values()) / (len(self.G) - 1)
        return avg_lengths

    #@timed
    def compute_avg_neighbor_degree(self):
        avg_neighbor_degree = {}
        for node in tqdm(self.subset_nodes):
            neighbors = list(self.G.neighbors(node))
            if neighbors:
                avg_neighbor_degree[node] = sum(self.G.degree(neighbor) for neighbor in neighbors) / len(neighbors)
            else:
                avg_neighbor_degree[node] = 0
        return avg_neighbor_degree

    #@timed
    def compute_local_reaching(self):
        def local_reaching_centrality(G, node):
            lengths = nx.single_source_shortest_path_length(G, node)
            reachable = len(lengths) - 1
            if reachable == 0:
                return 0
            return sum(1 / lengths[target] for target in lengths if target != node) / reachable

        return {node: local_reaching_centrality(self.G, node) for node in self.subset_nodes}

class ModelWeighted:

    def __init__(self, G, subset_nodes):
        """
        Initialize the ModelWeighted class with a graph and a subset of nodes.

        :param G: The input graph (a networkx graph)
        :param subset_nodes: A list or set of nodes to compute metrics for
        """
        self.G = G
        self.subset_nodes = subset_nodes

    def apply_edge_weights(self, PAE, weight=0.1):
        """
        Update the graph to include weights for risky edges.

        :param PAE: List of edges to apply the weight to
        :param weight: The weight to apply to the specified edges
        """
        for edge in PAE:
            if self.G.has_edge(edge[0], edge[1]):
                self.G[edge[0]][edge[1]]['weight'] = weight
        # return self.G

    #@timed
    def compute_weighted_in_degrees(self):
        return {node: self.G.in_degree(node, weight='weight') for node in self.subset_nodes}

    #@timed
    def compute_weighted_out_degrees(self):
        return {node: self.G.out_degree(node, weight='weight') for node in self.subset_nodes}

    #@timed
    def compute_weighted_betweenness(self):
        betweenness = nx.betweenness_centrality_subset(self.G, sources=self.subset_nodes, targets=self.subset_nodes, weight='weight')
        return {node: betweenness[node] for node in self.subset_nodes}

    def compute_weighted_harmonic(self):
        harmonic = {}
        for node in tqdm(self.subset_nodes):
            lengths = nx.single_source_dijkstra_path_length(self.G, node, weight='weight')
            harmonic[node] = sum(1 / length for target, length in lengths.items() if target != node)
        return harmonic

    #@timed
    def compute_weighted_eigenvector(self):
        eigenvector = nx.eigenvector_centrality_numpy(self.G, weight='weight')
        return {node: eigenvector[node] for node in self.subset_nodes}

    #@timed
    def compute_weighted_katz(self, alpha=0.1, beta=1.0, tol=1e-6, max_iter=1000):
        katz = nx.katz_centrality_numpy(self.G, alpha=alpha, beta=beta, weight='weight')

        min_bound = -1e6
        max_bound = 1e6
        katz = {node: max(min(katz[node], max_bound), min_bound) for node in katz}

        return {node: katz[node] for node in self.subset_nodes}

    #@timed
    def compute_weighted_pagerank(self):
        pagerank = nx.pagerank(self.G, weight='weight')
        return {node: pagerank[node] for node in self.subset_nodes}

    #@timed
    def compute_weighted_clustering(self):
        clustering = {node: nx.clustering(self.G, node, weight='weight') for node in self.subset_nodes}
        return clustering

    #@timed
    def compute_weighted_avg_shortest_path_length(self):
        avg_lengths = {}
        for node in tqdm(self.subset_nodes):
            lengths = nx.single_source_dijkstra_path_length(self.G, node, weight='weight')
            avg_lengths[node] = sum(lengths.values()) / (len(self.G) - 1)
        return avg_lengths

    #@timed
    def compute_weighted_avg_neighbor_degree(self):
        avg_neighbor_degree = nx.average_neighbor_degree(self.G, nodes=self.subset_nodes, weight='weight')
        return avg_neighbor_degree

    #@timed
    def compute_weighted_local_reaching(self):
        def local_reaching_centrality(G, node):
            lengths = nx.single_source_dijkstra_path_length(G, node, weight='weight')
            reachable = len(lengths) - 1
            if reachable == 0:
                return 0
            return sum(1 / lengths[target] for target in lengths if target != node) / reachable

        return {node: local_reaching_centrality(self.G, node) for node in self.subset_nodes}

def data_manager(strain,btrain,stest,btest,
                 in_degrees_subset,
                 out_degrees_subset,
                 betweenness_subset,
                 harmonic_subset,
                 eigenvector_subset,
                 katz_subset,
                 pagerank_subset,
                 clustering_subset,
                 avg_shortest_path_length_subset,
                 avg_neighbor_degree_subset,
                 local_reaching_subset):

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    test_index = []

    for node in strain + btrain:
        features = [
            in_degrees_subset[node],
            out_degrees_subset[node],
            betweenness_subset[node],
            harmonic_subset[node],
            eigenvector_subset[node],
            katz_subset[node],
            pagerank_subset[node],
            clustering_subset[node],
            avg_shortest_path_length_subset[node],
            avg_neighbor_degree_subset[node],
            local_reaching_subset[node]
        ]
        X_train.append(features)
        y_train.append(1 if node in strain else 0)

    for node in stest + btest:
        features = [
            in_degrees_subset[node],
            out_degrees_subset[node],
            betweenness_subset[node],
            harmonic_subset[node],
            eigenvector_subset[node],
            katz_subset[node],
            pagerank_subset[node],
            clustering_subset[node],
            avg_shortest_path_length_subset[node],
            avg_neighbor_degree_subset[node],
            local_reaching_subset[node]
        ]
        X_test.append(features)
        y_test.append(1 if node in stest else 0)
        test_index.append(node)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train,X_test,y_train,y_test,test_index

def evaluation(X_test,model,test_index,new_b=[]):
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    for i in range(len(test_index)):
        node = test_index[i]
        if node in new_b:
            y_pred[i]=0
            y_pred_proba[i]=0


    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    return accuracy,auc_score



for name1 in ['twitter+','Pokec+','facebook+', 'lastfm+']:
    for name2 in ['random', 'ba', 'bfs']:
    # for name2 in ['ba']:


        name = f'./datasets/{name1}{name2}/'
        print("\n\n", name1,name2)
        
        edges = load_txt_file_for_edges(name + 'edges.txt')
        PAE = load_txt_file_for_edges(name + 'PAE.txt')

        btrain = load_txt_file(name + 'btrain.txt')
        strain = load_txt_file(name + 'strain.txt')
        btest = load_txt_file(name + 'btest.txt')
        stest = load_txt_file(name + 'stest.txt')
        nodes = list(set(np.array(edges).reshape(-1)))
        new_b = load_txt_file(name + 'newly_benigns.txt')


        subset_nodes  = strain+stest+btrain+btest


        # start_time = time.time()
        # #############
        # end_time = time.time()
        # timer(start_time,end_time)

        start_time = time.time()
        G = nx.DiGraph()
        G.add_edges_from(edges)
        model = ModelUnweighted(G,subset_nodes)

        in_degrees_subset = model.compute_in_degrees()
        out_degrees_subset = model.compute_out_degrees()
        betweenness_subset = model.compute_betweenness()
        harmonic_subset = model.compute_harmonic()
        eigenvector_subset = model.compute_eigenvector()
        katz_subset = model.compute_katz()
        pagerank_subset = model.compute_pagerank()
        clustering_subset = model.compute_clustering()
        avg_shortest_path_length_subset = model.compute_avg_shortest_path_length()
        avg_neighbor_degree_subset = model.compute_avg_neighbor_degree()
        local_reaching_subset = model.compute_local_reaching()

        X_train,X_test,y_train,y_test,test_index = data_manager(strain,
                                                                btrain,
                                                                stest,
                                                                btest,
                                                                in_degrees_subset,
                                                                out_degrees_subset,
                                                                betweenness_subset,
                                                                harmonic_subset,
                                                                eigenvector_subset,
                                                                katz_subset,
                                                                pagerank_subset,
                                                                clustering_subset,
                                                                avg_shortest_path_length_subset,
                                                                avg_neighbor_degree_subset,
                                                                local_reaching_subset)



        # Initialize and train the logistic regression model
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)

        end_time = time.time()
        eshtrak = end_time - start_time 

        start_time = time.time()

        print("init")
        accuracy,auc_score = evaluation(X_test,model,test_index,new_b=[])
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"AUC Score: {auc_score:.2f}")

        end_time = time.time()
        timer(start_time,end_time,eshtrak)
        start_time = time.time()

        print("MB")
        accuracy,auc_score = evaluation(X_test,model,test_index,new_b=new_b)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"AUC Score: {auc_score:.2f}")

        end_time = time.time()
        timer(start_time,end_time,eshtrak)
        start_time = time.time()


        G = nx.DiGraph()
        G.add_edges_from(edges)
        model = ModelWeighted(G, subset_nodes)
        model.apply_edge_weights(PAE, weight=0.01)

        in_degrees_subset = model.compute_weighted_in_degrees()
        out_degrees_subset = model.compute_weighted_out_degrees()
        betweenness_subset = model.compute_weighted_betweenness()
        harmonic_subset = model.compute_weighted_harmonic()
        eigenvector_subset = model.compute_weighted_eigenvector()
        katz_subset = model.compute_weighted_katz()
        pagerank_subset = model.compute_weighted_pagerank()
        clustering_subset = model.compute_weighted_clustering()
        avg_shortest_path_length_subset = model.compute_weighted_avg_shortest_path_length()
        avg_neighbor_degree_subset = model.compute_weighted_avg_neighbor_degree()
        local_reaching_subset = model.compute_weighted_local_reaching()



        X_train,X_test,y_train,y_test,test_index = data_manager(strain,
                                                                btrain,
                                                                stest,
                                                                btest,
                                                                in_degrees_subset,
                                                                out_degrees_subset,
                                                                betweenness_subset,
                                                                harmonic_subset,
                                                                eigenvector_subset,
                                                                katz_subset,
                                                                pagerank_subset,
                                                                clustering_subset,
                                                                avg_shortest_path_length_subset,
                                                                avg_neighbor_degree_subset,
                                                                local_reaching_subset)


        # Initialize and train the logistic regression model
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)

        end_time = time.time()
        
        eshtrak = end_time - start_time 
        
        start_time = time.time()
        

        print("PAE")
        accuracy,auc_score = evaluation(X_test,model,test_index,new_b=[])
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"AUC Score: {auc_score:.2f}")

        end_time = time.time()
        timer(start_time,end_time,eshtrak)
        start_time = time.time()
        

        print("PAE+MB")
        accuracy,auc_score = evaluation(X_test,model,test_index,new_b=new_b)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"AUC Score: {auc_score:.2f}")

        end_time = time.time()
        timer(start_time,end_time,eshtrak)
        start_time = time.time()
        