""More-Efficient-Sybil-Detection-Mechanisms-Leveraging-Resistance-of-Users-to-Attack-Requests""


# Instructions for Running the Code

## 1. Run `synthesize_dataset.py`
- This script generates datasets based on three different attack strategies.
- Visualization using `Drawer` is commented out in the code due to time constraints for large graphs, but you can enable it if needed.
- **Input Format**: A text file named `edge.txt` containing `m` lines, where each line represents an edge with two integers corresponding to the nodes at each end. Note that edges are assumed to be directed. If your dataset is undirected, you must include both directions for each edge.
- **Sample Input**: Use the files provided in the "just edges semi-raw dataset.zip" or the sample `edge.txt` file in the root directory.
- **Output**: A zip file is created, containing the necessary files for subsequent steps. Directories should be manually created as required.

## 2. Run `train-extractor.py`
- This script is used to randomly split the dataset into training and testing sets.

## 3. Run `probability_generator.py`
- This script assigns resistance probabilities to nodes in the graph.

## 4. Run `mb+pae.py`
- This script runs both MB and PAE preprocessing simultaneously.
- For the MB method, a traversing algorithm is used in this file.

## 5. Run `PAE.py`
- This script is specifically for visualizing the PAE experiment results.

## 6. Run `mb_experiments_visualization.py`
- This script provides visualizations for different MB algorithms and their performance.
- In this code you can change range to see outputs faster, in bigger range, the runtime will be much longer.
- In this code, title and dataset is chosen by input function.

## 7. Run `attack_statistics.py`
- This script generates the statistics used in the related table in the paper.

## 8. Models Used in the Paper
- The three main models used are:
  1. `sybilSCAR.py`
  2. `sybilWalk.py`
  3. `sybilmetric.py`
  
- **Note**: `sybilmetric.py` is more time-intensive compared to the other two methods.


## Most Important Libraries

### Name: `networkx`
- Version: 3.2.1

### Name: `numpy`
- Version: 1.21.5

### Name: `matplotlib`
- Version: 3.5.2
