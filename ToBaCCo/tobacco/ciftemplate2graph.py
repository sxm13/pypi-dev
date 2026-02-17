from __future__ import print_function
import re
import os
import numpy as np
import networkx as nx
from bbcif_properties import bb2lattices, bb2labels, bb2symbols, bb2coords, bb2bonds

def ct2g(path):
    
	cifname = os.path.basename(path)
	
	lattice, unit_cell = bb2lattices(path)
	aL, bL, cL, alpha, beta, gamma = lattice

	G = nx.MultiGraph()

	nc = 0
	ne = 0
	types = []
	max_le = 1.0e6

	label_loop = bb2labels(path)
	type_loop = bb2symbols(path)
	ccoords, fcoords = bb2coords(path)
	ccoords_T, fcoords_T = np.array(ccoords).T, np.array(fcoords).T
	for idx, label in enumerate(label_loop):
		label = label_loop[idx]
		atom_type = type_loop[idx]

		nc += 1
		types.append(atom_type)
		
		G.add_node(
			label,
			type=atom_type,
			index=nc,
			ccoords=ccoords_T[idx],
			fcoords=fcoords_T[idx],
			cn=[],
			cifname=[]
		)

	bond_label1_loop, bond_label2_loop, bond_dis_loop, bond_sym_loop, _ = bb2bonds(path)

	added_edges = set()

	for idx, bond_label1 in enumerate(bond_label1_loop):

		label1 = bond_label1
		label2 = bond_label2_loop[idx]
		length = float(bond_dis_loop[idx])
		symm = bond_sym_loop[idx]

		if symm == '.':
			lbl = np.array([0, 0, 0])
		elif '_' in symm:
			lbl = lbl = np.asarray(list(map(int, symm.split('_')[1]))) - 5
		else:
			raise ValueError('Error in ciftemplate2graph, there are unrecognized bond translational symmetries in' + cifname)

		nlbl = -1 * lbl

		edge_tuple = (label1, label2, lbl[0], lbl[1], lbl[2])
		edge_tuple_rev1 = (label2, label1, lbl[0], lbl[1], lbl[2])
		edge_tuple_rev2 = (label1, label2, nlbl[0], nlbl[1], nlbl[2])
		edge_tuple_rev3 = (label2, label1, nlbl[0], nlbl[1], nlbl[2])
		if edge_tuple in added_edges or edge_tuple_rev1 in added_edges or edge_tuple_rev2 in added_edges or edge_tuple_rev3 in added_edges:
			continue
		added_edges.add(edge_tuple)
		if label1 not in G.nodes or label2 not in G.nodes:
			continue

		ne += 1

		v1 = G.nodes[label1]['fcoords']
		v2 = G.nodes[label2]['fcoords'] + lbl

		ef_coords = np.average(np.array([v1, v2]), axis=0)
		ec_coords = np.dot(unit_cell, ef_coords)

		cdist = np.linalg.norm(np.dot(unit_cell, v1 - v2))

		if cdist < max_le:
			max_le = cdist

		G.add_edge(
			label1,
			label2,
			key=(ne, lbl[0], lbl[1], lbl[2]),
			label=lbl,
			length=length,
			fcoords=ef_coords,
			ccoords=ec_coords,
			index=ne,
			pd=(label1, label2)
		)

	if G.number_of_edges() == 0:
		raise ValueError('Error in ciftemplate2graph, no edges are given in the template:',cifname)

	S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
	catenation = len(S) > 1

	for net in [(s, unit_cell, cifname, aL, bL, cL, alpha, beta, gamma, max_le) for s in S]:

		SG = nx.MultiGraph()
		cns = []
		e_types = []

		count = 0
		start = None

		for node in net[0].nodes(data=True):
			n, data = node
			cn = G.degree(n)
			ty = re.sub('[0-9]', '', n)
			cns.append((cn, ty))
			SG.add_node(n, **data)
			
			if count == 0:
				start = data['fcoords']
			count += 1

		count = 0
		for edge in net[0].edges(keys=True, data=True):
			count += 1
			e0, e1, key, data = edge
			key = tuple([count] + [k for k in key[1:]])
			data['index'] = count
			
			l = sorted([re.sub('[^a-zA-Z]', '', e0), re.sub('[^a-zA-Z]', '', e1)])
			e_types.append((l[0], l[1]))
			
			SG.add_edge(e0, e1, key=key, type=(l[0], l[1]), **data)

	yield (
		SG, start, unit_cell, set(cns), set(e_types), cifname,
		aL, bL, cL, alpha, beta, gamma, max_le, catenation
	)


def node_vecs(node, G, unit_cell, label):

	edge_coords = []
	edge_coords_append = edge_coords.append

	for e in G.edges(data=True):

		edict = e[2]
		positive_direction = edict['pd']
		lbl = edict['label']
		ind = edict['index']

		if node in e[0:2]:

			if node == positive_direction[0]:
				vec = edict['ccoords']
			else:
				vec = np.dot(unit_cell, -1 * lbl + edict['fcoords'])
			if label:
				edge_coords_append([ind, vec])
			else:
				edge_coords_append(vec)

	if label:
		ec_com = np.average(np.asarray([v[1] for v in edge_coords]), axis=0)
	else:
		ec_com = np.average(edge_coords, axis = 0)
	if label:
		shifted_edge_coords = [[v[0], v[1] - ec_com] for v in edge_coords]
	else:
		shifted_edge_coords = [vec - ec_com for vec in edge_coords]

	return shifted_edge_coords

def edge_vecs(edge, G):

	for e in G.edges(data=True):
		edict = e[2]
		if edict['index'] == edge:
			s,e = e[0:2]
			ccoords = edict['ccoords']
			v1 = G.nodes[s]['ccoords']
			v2 = G.nodes[e]['ccoords']

			return [v1 - ccoords, v2 - ccoords]
