import numpy as np
import networkx as nx
import re


def omega2coords(start, TG, sc_omega_plus, num_vertices):

	shortest_path_dict = nx.shortest_path(TG)
	SN = sorted(TG.nodes(), key = lambda x : int(re.sub('[A-Za-z]','',x)))
	sequential_paths = [(SN[i],SN[i+1]) for i in range(num_vertices) if i+1 < num_vertices]
	start = np.asarray(start)

	cnd = nx.get_node_attributes(TG, 'cifname')
	coords = [[sequential_paths[0][0], cnd[sequential_paths[0][0]], start, [(e[2]['index'],e[2]['pd'], e[2]['cifname']) for e in TG.edges(data=True) if sequential_paths[0][0] in e]]]
	coords_append = coords.append
	already_placed = [sequential_paths[0][0]]
	already_placed_append = already_placed.append

	for st in sequential_paths:

		shortest_path_dict = dict(nx.all_pairs_shortest_path(TG))
		st_path = shortest_path_dict[st[0]][st[1]]
		lp = len(st_path)
		traverse = [(st_path[i], st_path[i+1]) for i in range(lp) if i+1 < lp]

		for e0 in traverse:

			s,e = e0
			edict = TG[s][e]
			key = [k for k in edict][0]
			ind = key[0]
			positive_direction = edict[key]['pd']

			if (s,e) == positive_direction:
				direction = 1
			elif (e,s) == positive_direction:
				direction = -1
			else:
				raise ValueError('Error in defining an edge traversal in omega_to_coords.py')
			start = start + direction * sc_omega_plus[ind - 1]

			if e0[1] not in already_placed:
				coords_append([e0[1], cnd[e0[1]], start, [(e[2]['index'], e[2]['pd'],e[2]['cifname']) for e in TG.edges(data=True) if e0[1] in e]])
				already_placed_append(e0[1])
	
	norm_coords = []
	norm_coords_append = norm_coords.append
	for line in coords:
		vec = []
		vec_append = vec.append
		v = line[0]
		for dim in line[2]:
			if dim < 0.0:
				while dim < 0:
					dim = dim + 1.0
			elif dim >= 1.0:
				while dim >= 1.0:
					dim = dim - 1.0
			vec_append(dim)
		norm_coords_append([line[0],line[1],vec,line[3]])

	norm_coords = sorted(norm_coords, key = lambda x : int(re.sub('[A-Za-z]', '', x[0])))

	return norm_coords
