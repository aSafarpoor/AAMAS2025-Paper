import os 
import random
# random.seed(1)
import json

# 1 − r(v))x3 + r(v)(1 − x3)

def load_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        return {int(float(k)): int(float(v)) for k, v in data.items()}


    
probability_of_resistance = {}


for name1 in ['Pokec+','facebook+','twitter+','lastfm+']:
	for name2 in ['random', 'ba', 'bfs']:

		name = f'./datasets/{name1}{name2}/'
		
		r = load_json_file(os.path.join(name, 'resistanceDictionary.json'))


		for node in r.keys():
			x = random.random()
			if r[node] == 1:
				probability_of_resistance[node] = 1 - x**3
			else:
				probability_of_resistance[node] = x**3

		filename = f"{name}probability_of_resistance_dictionary.json"
		with open(filename, 'w') as file:
		    json.dump(probability_of_resistance, file, indent=4)

		print(name,'done')