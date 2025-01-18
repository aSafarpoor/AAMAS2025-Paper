

import heapq
import numpy as np
from tqdm import tqdm
import random
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import time

# Function to measure and print the elapsed time in milliseconds between start_time and end_time
def timer(start_time,end_time):
	elapsed_time_s = end_time - start_time
	elapsed_time_ms = elapsed_time_s * 1000
	print(f"Elapsed time in milliseconds: {elapsed_time_ms:.6f} ms")

# Function to load a text file where each line contains an integer
def load_txt_file(filename):
	with open(filename, 'r') as file:
		return [int(line.strip()) for line in file]

# Function to load a text file for edges where each line contains two integers (an edge between two nodes)
def load_txt_file_for_edges(filename):
	with open(filename, 'r') as file:
		return [list(map(int,line.strip().split())) for line in file]

# Class for Simple Message Passing Graph Propagation
class Simple_MSG_Propagation:
	def __init__(self, nodes, edges, btrain, btest, strain, stest, pae=[]):
		# Initialize the graph parameters
		self.nodes = nodes  # List of nodes in the graph
		self.edges = edges  # List of edges in the graph
		self.pmatrix = np.zeros(len(self.nodes))  # Probability matrix for node values
		self.pmatrix = self.pmatrix.reshape((-1, 1))  # Reshape to a column vector

		# Initialize adjacency matrix A for propagation
		self.A = np.zeros((len(nodes)+2, len(nodes)+2))  # Extra rows/columns for benign/sybil labels
		pae_set = set(tuple(edge) for edge in pae)  # Set of pseudo edges (PAE)

		# Fill in adjacency matrix based on edges
		for e in edges:
			e = tuple(e)
			self.A[e[0], e[1]] = 1  # Normal edge weight

			if e in pae_set:
				self.A[e[0], e[1]] = 0.1  # Weight for pseudo edges

		# Connect benign and sybil nodes to special source nodes
		for node in btrain:
			self.A[-1, node] = 1  # Connect benign training nodes

		for node in strain:
			self.A[-2, node] = 1  # Connect sybil training nodes

		# Normalize rows of adjacency matrix
		row_sums = self.A.sum(axis=0)  # Sum of each row (out-degrees)

		for i in range(len(self.A)):
			if row_sums[i] > 0:
				self.A[i, :] /= row_sums[i]  # Normalize rows with non-zero sums

		# Store train and test sets
		self.btrain = btrain  # Benign train nodes
		self.btest = btest  # Benign test nodes
		self.strain = strain  # Sybil train nodes
		self.stest = stest  # Sybil test nodes

	# Message propagation algorithm
	def msg_prop(self, sybilval, beningval, rounds=5, alpha=1, unknownval=0):
		self.rounds = rounds  # Number of propagation rounds
		self.alpha = alpha  # Smoothing factor
		pmatrix = np.ones(len(self.nodes) + 2) * unknownval  # Initialize probability matrix
		pmatrix = pmatrix.reshape((-1, 1))  # Reshape to column vector
		value_list = []
		value_list.append(pmatrix[:])  # Save initial state of the matrix

		# Assign initial values for benign and sybil training nodes
		for x in self.btrain:
			pmatrix[x] = beningval
		for x in self.strain:
			pmatrix[x] = sybilval

		# Perform message passing for a set number of rounds
		for i in range(rounds):
			pnew = np.dot(self.A, pmatrix)  # Matrix multiplication for message propagation
			pmatrix = (1 - alpha) * pmatrix + alpha * np.copy(pnew)  # Weighted update with alpha
			pmatrix[-1] = beningval  # Maintain benign value at source node
			pmatrix[-2] = sybilval  # Maintain sybil value at source node
			value_list.append(pmatrix[:])  # Save updated probabilities

		return value_list  # Return the list of probability matrices

	# Testing the performance using AUC score
	def test(self, value_list, th, extra=[]):
		btest = self.btest  # Benign test nodes
		stest = self.stest  # Sybil test nodes
		output = np.array(value_list)[-1].reshape(-1)  # Get final probability values
		label = []
		prediction = []
		extra = set(extra)  # Extra benign nodes to exclude from sybil detection

		# Generate predictions for benign test nodes
		for i in btest:
			if i in extra:
				x = -1  # Ignore certain nodes in extra
			else:
				x = output[i]

			prediction.append((x + 1) / 2)  # Normalize predictions to [0, 1] range
			label.append(0)  # Benign label is 0

		# Generate predictions for sybil test nodes
		for i in stest:
			x = output[i]
			prediction.append((x + 1) / 2)  # Normalize predictions to [0, 1] range
			label.append(1)  # Sybil label is 1

		# Calculate and print AUC score
		auc = roc_auc_score(label, prediction)
		print("AUC= ", auc)

# Load datasets and run the propagation algorithm
for name1 in ['twitter+', 'Pokec+', 'facebook+', 'lastfm+']:
	for name2 in ['random', 'ba', 'bfs']:

		name = f'./datasets/{name1}{name2}/'  # Dataset path
		print("\n\n", name1, name2)

		# Load dataset files for edges, pseudo-edges (PAE), and train/test nodes
		edges = load_txt_file_for_edges(name + 'edges.txt')
		pae_edges = load_txt_file_for_edges(name + 'PAE.txt')

		btrain = load_txt_file(name + 'btrain.txt')
		strain = load_txt_file(name + 'strain.txt')
		btest = load_txt_file(name + 'btest.txt')
		stest = load_txt_file(name + 'stest.txt')
		nodes = list(set(np.array(edges).reshape(-1)))  # Extract unique nodes from edge list
		btrain2 = load_txt_file(name + 'newly_benigns.txt')  # Load extra benign nodes

		# Test message propagation with different setups and configurations
		start_time = time.time()
		print("init")
		smsgp = Simple_MSG_Propagation(nodes, edges, btrain, btest, strain, stest)
		values = smsgp.msg_prop(rounds=4, alpha=0.3, sybilval=1, beningval=-1, unknownval=0)
		smsgp.test(value_list=values, th=0)
		end_time = time.time()
		timer(start_time, end_time)

