from __future__ import print_function
import numpy as np
import re, os
from itertools import combinations
import datetime
import networkx as nx
from utils import PBC3DF, PBC3DF_sym, nn, nl
import gemmi
from bbcif_properties import bb2lattices, bb2labels, bb2coords, bb2bonds


def distance_search_bond(placed_all, bonds_all, sc_unit_cell, tol):

	fixed_bonds = []
	used_bonds = []
	fixed_bonds_append = fixed_bonds.append
	used_bonds_append = used_bonds.append
	
	for l in bonds_all:
		fixed_bonds_append([l[0], l[1], l[2], '.', l[4]])
		used_bonds_append((l[0], l[1]))

	connection_points = [line for line in placed_all if re.sub('[0-9]', '', line[5]) == 'X']
	nbcount = 0
			
	for i in range(len(connection_points)):

		ielem = connection_points[i][0]
		ivec = np.dot(np.linalg.inv(sc_unit_cell), np.array([float(q) for q in connection_points[i][1:4]]))
		ibbid = int(connection_points[i][6])

		for j in range(i + 1, len(connection_points)):

			jelem = connection_points[j][0]
			jbbid = int(connection_points[j][6])

			if (ielem, jelem) not in used_bonds and (jelem, ielem) not in used_bonds:

				jvec = np.dot(np.linalg.inv(sc_unit_cell), list(map(float, connection_points[j][1:4])))
				DV, sym = PBC3DF_sym(ivec, jvec, "write_cifs'")
				dist = np.linalg.norm(np.dot(sc_unit_cell, DV))
	
				if dist < tol and ibbid != jbbid:
					nbcount += 1
					fixed_bonds_append([ielem, jelem, dist, sym, 'S'])
					break
				else:
					continue

	return fixed_bonds, nbcount

def bond_connected_components(placed_all, bonds_all, sc_unit_cell, max_length, bond_tol,
							  nconnections_list, num_possible_XX_bonds):

	G = nx.Graph()

	for n in placed_all:
		isX = (re.sub('[0-9]', '', n[-3]) == 'X')
		G.add_node(n[0], coords=np.array(list(map(float,(n[1:4])))), occ=n[4], isX=isX, bbcode=int(n[5]),
			 														 nconnect=1, bbtype=n[-1])
	for l in bonds_all:
		G.add_edge(l[0], l[1], length=l[2], sym=l[3], ty=l[4], is_new_XX=False, order=(l[0],l[1]))

	for line in nconnections_list:
		node,nc = line
		G.nodes[node]['nconnect'] -= 1
		G.nodes[node]['nconnect'] += nc

	previous_degrees = dict((node, G.degree(node)) for node in G.nodes())

	ccs = list(nx.connected_components(G))
	count = 0
	bb_tol = max_length + 0.50*max_length
	print('distance search tolerance is', np.round(bb_tol,3), 'Angstroms')

	for connect_comp0, connect_comp1 in combinations(ccs, 2):

		xname0 = [n for n in connect_comp0 if G.nodes[n]['isX']]
		xvecs0 = [np.dot(np.linalg.inv(sc_unit_cell),
				  G.nodes[n]['coords']) for n in connect_comp0 if G.nodes[n]['isX']]
		com0 = np.average(xvecs0, axis=0)
		type0 = list(set([G.nodes[n]['bbtype'] for n in connect_comp0]))

		xname1 = [n for n in connect_comp1 if G.nodes[n]['isX']]
		xvecs1 = [np.dot(np.linalg.inv(sc_unit_cell),
				  G.nodes[n]['coords']) for n in connect_comp1 if G.nodes[n]['isX']]
		com1 = np.average(xvecs1, axis=0)
		type1 = list(set([G.nodes[n]['bbtype'] for n in connect_comp1]))

		if len(type0) > 1 or len(type1) > 1:
			print(type0, type1)
			raise ValueError('building block indicated as both node and edge type')

		if len(xname0) == 0 or len(xname1) == 0:
			raise ValueError('There are connected components with no connection site atoms')

		type0 = type0[0]
		type1 = type1[0]
		# don't want to try any edge-edge bonds, this will always result in incorrect bonding that must be corrected later
		if type0 == 'edge' and type1 == 'edge':
			continue

		com_dist = np.linalg.norm(np.dot(sc_unit_cell, com0 - PBC3DF(com0,com1)))

		if com_dist <= bb_tol:

			min_dist = (100.0, 'no.3', 'the', 'larch')

			for xv0,xn0 in zip(xvecs0, xname0):
				for xv1,xn1 in zip(xvecs1, xname1):

					DV, sym = PBC3DF_sym(xv0, xv1, "write_cifs")
					dist = np.linalg.norm(np.dot(sc_unit_cell, DV))

					if dist < min_dist[0]:
						min_dist = (dist, xn0, xn1, sym)

			if min_dist[0] < bond_tol:
				count += 1
				G.add_edge(min_dist[1], min_dist[2], length=min_dist[0],
			               sym=min_dist[3], ty='S', is_new_XX=True, order=(min_dist[1],min_dist[2]))

	# attempt to bond any connection sites left without X neighbors
	# by forming bonds (distance search) within the X atoms withoug any X neighbors
	connection_nodes = [(n, data) for n, data in G.nodes(data=True) if data['isX']]
	no_bond_connection_sites = []
	
	for node,data in connection_nodes:

		bbcode0 = data['bbcode']
		nconnections = data['nconnect']
		nbors = list(G.neighbors(node))
		X_nbors = [nbor for nbor in nbors if (G.nodes[nbor]['isX'] and G.nodes[nbor]['bbcode'] != bbcode0)]

		if len(X_nbors) < nconnections:
			no_bond_connection_sites.append(node)

	for n0,n1 in combinations(no_bond_connection_sites, 2):
		
		vec0 = np.dot(np.linalg.inv(sc_unit_cell), G.nodes[n0]['coords'])
		vec1 = np.dot(np.linalg.inv(sc_unit_cell), G.nodes[n1]['coords'])
	
		bbcode0 = G.nodes[n0]['bbcode']
		bbcode1 = G.nodes[n1]['bbcode']
	
		bbtype0 = G.nodes[n0]['bbtype']
		bbtype1 = G.nodes[n1]['bbtype']
	
		if bbcode0 == bbcode1:
			continue
		if bbtype0 == 'edge' and bbtype1 == 'edge':
			continue
	
		DV, sym = PBC3DF_sym(vec0,vec1, "write_cifs")
		dist = np.linalg.norm(np.dot(sc_unit_cell, DV))
	
		if dist < bond_tol:
			count += 1
			G.add_edge(n0, n1, length=dist, sym=sym, ty='S', is_new_XX=True, order=(n0,n1))

	# remove any extra bonds by keeping only bonds to the closest N X-neighbors
	# where N is the number of possible connections for each X
	for node, data in connection_nodes:
		
		vec0 = np.dot(np.linalg.inv(sc_unit_cell), G.nodes[node]['coords'])
		bbcode0 = data['bbcode']
		bbtype0 = data['bbtype']
	
		nbors = list(G.neighbors(node))
		X_nbors = [nbor for nbor in nbors if (G.nodes[nbor]['isX'] and G.nodes[nbor]['bbcode'] != bbcode0)]
		nconnections = data['nconnect']
	
		if len(X_nbors) > nconnections:
	
			nbor_dists = []
			for nbor in X_nbors:
				vec1 = np.dot(np.linalg.inv(sc_unit_cell), G.nodes[nbor]['coords'])
				dist = np.linalg.norm(np.dot(sc_unit_cell, vec0 - PBC3DF(vec0,vec1)))
				nbor_dists.append((dist,nbor))
			nbor_dists.sort(key=lambda x:x[0])
	
			for nbor in nbor_dists[nconnections:]:
				G.remove_edge(node, nbor[1])
				count -= 1

	wrong_connection_nodes = []
	wrong_connection_nodes_append = wrong_connection_nodes.append

	for node,data in connection_nodes:

		nconnect = data['nconnect']
		nbors = list(G.neighbors(node))
		bbcode0 = data['bbcode']

		X_nbors = [nbor for nbor in nbors if (G.nodes[nbor]['isX'] and G.nodes[nbor]['bbcode'] != bbcode0)]
		if len(X_nbors) != nconnect:
			wrong_connection_nodes_append(node)

	XX_bond_count = 0
	for _, _, data in G.edges(data=True):
		if data['is_new_XX']:
			XX_bond_count += 1
	
	if XX_bond_count != count:
		raise ValueError('bond counts do not match, there is a problem with the connection site bonding algorithm')

	bond_check_passed = True

	for node,data in G.nodes(data=True):

		isX = data['isX']
		degree = G.degree(node)
		previous_degree = previous_degrees[node]
		nconnections = data['nconnect']

		if isX and degree != previous_degree + nconnections:
			print('There are connection sites with too many/not enough bonds formed.')
			bond_check_passed = False
			break
		elif not isX and degree != previous_degree:
			print('The degree of a non-X atom has changed after XX bond formation.')
			bond_check_passed = False
			break

	if len(wrong_connection_nodes) > 0:
		print('There are connection sites with too few or too many bonds')
		bond_check_passed = False
	if XX_bond_count < num_possible_XX_bonds:
		print('The number of XX bonds formed',
				XX_bond_count, 'is less than the correct number', num_possible_XX_bonds)
		print('Try increasing the bond formation distance tolerance.')
		bond_check_passed = False
	if XX_bond_count > num_possible_XX_bonds:
		print('The number of XX bonds formed', XX_bond_count,
		      'is more than the correct number', num_possible_XX_bonds)
		print('Check your inputs for incorrect X atoms.')
		bond_check_passed = False

	fixed_bonds = []
	fixed_bonds_append = fixed_bonds.append

	for edge in G.edges(data=True):
		
		edict = edge[2]
		ty = edict['ty']
		leng = edict['length']
		sy = edict['sym']
		order = edict['order']
		fixed_bonds_append([order[0], order[1], leng, sy, ty])

	return fixed_bonds, count, bond_check_passed

def fix_bond_sym(bonds_all, placed_all, sc_unit_cell):
	
	coords_dict = dict((l[0], np.dot(np.linalg.inv(sc_unit_cell), list(map(float, l[1:4])))) for l in placed_all)

	fixed_bonds = []
	fixed_bonds_append = fixed_bonds.append
	for l in bonds_all:
		vec1 = coords_dict[l[0]]
		vec2 = coords_dict[l[1]]

		dist,sym = PBC3DF_sym(vec1,vec2, "write_cifs")

		fixed_bonds_append([l[0], l[1], l[2], sym, l[4]])

	return fixed_bonds

def write_cif(placed_all, fixed_bonds, scaled_params, sc_unit_cell, opath, wrap_coords=True):
    sc_a, sc_b, sc_c, sc_alpha, sc_beta, sc_gamma = scaled_params

    doc = gemmi.cif.Document()
    block = doc.add_new_block("data_" + opath.split("/")[-1][:-4])

    block.set_pair("_symmetry_space_group_name_H-M", "P1")
    block.set_pair("_symmetry_Int_Tables_number", "1")
    block.set_pair("_symmetry_cell_setting", "triclinic")
    block.set_pair("_cell_length_a", str(sc_a))
    block.set_pair("_cell_length_b", str(sc_b))
    block.set_pair("_cell_length_c", str(sc_c))
    block.set_pair("_cell_angle_alpha", str(sc_alpha))
    block.set_pair("_cell_angle_beta", str(sc_beta))
    block.set_pair("_cell_angle_gamma", str(sc_gamma))

    sym_loop = block.init_loop("_symmetry_equiv_pos_as_xyz", ["_symmetry_equiv_pos_as_xyz"])
    sym_loop.add_row(["x,y,z"])

    atom_loop = block.init_loop("_atom_site_label", [
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ])

    inv_cell = np.linalg.inv(np.array(sc_unit_cell, dtype=float))

    for l in placed_all:
        lab = l[0]
        sym = re.sub(r"[0-9]", "", lab)
        vec = np.array(list(map(float, l[1:4])), dtype=float)
        f = inv_cell @ vec
        if wrap_coords:
            f = np.mod(f, 1.0)
        atom_loop.add_row([lab, sym, f"{f[0]:.10f}", f"{f[1]:.10f}", f"{f[2]:.10f}"])

    if fixed_bonds:
        bond_loop = block.init_loop("_geom_bond_atom_site_label_1", [
            "_geom_bond_atom_site_label_1",
            "_geom_bond_atom_site_label_2",
            "_geom_bond_distance",
            "_geom_bond_site_symmetry_2",
            "_ccdc_geom_bond_type",
        ])
        for e in fixed_bonds:
            bond_loop.add_row([str(e[0]), str(e[1]), f"{float(e[2]):.3f}", str(e[3]), str(e[4])])

    doc.write_file(opath)


def read4merge(filename):

	lattice, _ = bb2lattices(filename)
	a, b, c, alpha, beta, gamma = lattice

	names = bb2labels(filename)

	_, fcoords_ori = bb2coords(filename)
	fcoords = []
	for fcoord in fcoords_ori:
		fvec = np.array([np.round(float(v),8) for v in fcoord])
		for dim in range(len(fvec)):
			if fvec[dim] < 0.0:
				fvec[dim] += 1.0
			elif fvec[dim] > 1.0:
				fvec[dim] -= 1.0

		fcoords.append(fvec)
	fcoords = np.asarray(fcoords)

	bond_label1_loop, bond_label2_loop, bond_dis_loop, bond_sym_loop, bond_type_loop = bb2bonds(filename)
	bonds = []
	for idx in range(len(bond_type_loop)):
		bonds.append((  bond_label1_loop[idx],
						bond_label2_loop[idx],
						bond_dis_loop[idx],
						bond_sym_loop[idx],
						bond_type_loop[idx]))
	
	return names, fcoords, bonds, (a,b,c,alpha,beta,gamma)


def merge_catenated_cifs(comb):
	
	all_read = [read4merge(cif) for cif in comb]
	natoms = [len(c[0]) for c in all_read]
	a, b, c, alpha, beta, gamma = all_read[0][3]
	
	if len(set(natoms)) != 1:
		raise ValueError('The number of atoms is not the same for the cifs:', comb)

	cifname = comb[0][0:-9] + '.cif'
	blockname = 'data_' + os.path.basename(cifname)[0:-4]

	doc = gemmi.cif.Document()
	block = doc.add_new_block(blockname)

	block.set_pair('_audit_creation_date', datetime.datetime.today().strftime('%Y-%m-%d'))
	block.set_pair('_audit_creation_method', 'tobacco_3.0')
	block.set_pair("_symmetry_space_group_name_H-M", 'P1')
	block.set_pair('_symmetry_Int_Tables_number', '1')
	block.set_pair('_symmetry_cell_setting', 'triclinic')

	sym_loop = block.init_loop('_symmetry_equiv_pos_as_xyz', ['_symmetry_equiv_pos_as_xyz'])
	sym_loop.add_row(['x,y,z'])

	block.set_pair('_cell_length_a', str(a))
	block.set_pair('_cell_length_b', str(b))
	block.set_pair('_cell_length_c', str(c))
	block.set_pair('_cell_angle_alpha', str(alpha))
	block.set_pair('_cell_angle_beta', str(beta))
	block.set_pair('_cell_angle_gamma', str(gamma))

	atom_loop = block.init_loop('_atom_site_label', [
		'_atom_site_label',
		'_atom_site_type_symbol',
		'_atom_site_fract_x',
		'_atom_site_fract_y',
		'_atom_site_fract_z',
	])

	count = 0
	n0 = natoms[0]
	for cif in all_read:
		names, fcoords, bonds, cellpar = cif
		shift = count * n0
		count += 1

		new_names = [nn(name) + str(int(nl(name)) + shift) for name in names]
		for name, coord in zip(new_names, fcoords):
			atom_loop.add_row([
				name,
				nn(name),
				f"{float(coord[0]):.8f}",
				f"{float(coord[1]):.8f}",
				f"{float(coord[2]):.8f}",
			])

	bond_loop = block.init_loop('_geom_bond_atom_site_label_1', [
		'_geom_bond_atom_site_label_1',
		'_geom_bond_atom_site_label_2',
		'_geom_bond_distance',
		'_geom_bond_site_symmetry_2',
		'_ccdc_geom_bond_type',
	])

	count = 0
	for cif in all_read:
		names, fcoords, bonds, cellpar = cif
		shift = count * n0
		count += 1

		new_bonds = [
			(
				nn(b0) + str(int(nl(b0)) + shift),
				nn(b1) + str(int(nl(b1)) + shift),
				f"{float(dist):.3f}",
				str(sym2),
				str(btype),
			)
			for (b0, b1, dist, sym2, btype) in bonds
		]

		for row in new_bonds:
			bond_loop.add_row(list(row))

	doc.write_file(cifname)