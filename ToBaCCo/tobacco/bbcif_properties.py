import re
import numpy as np
from utils import PBC3DF
import gemmi
from ase.io import read


def bb2lattices(path):

    atoms = read(path)
    cell = atoms.cell
    aL, bL, cL = cell.lengths()
    alpha, beta, gamma = cell.angles()
    aL,bL,cL,alpha,beta,gamma = list(map(float, (aL,bL,cL,alpha,beta,gamma)))
    unit_cell = cell.array.T

    return (aL,bL,cL,alpha,beta,gamma), unit_cell

def bb2labels(path):

    cif_doc = gemmi.cif.read_file(path)
    block = cif_doc[0]
    label_loop = block.find_loop('_atom_site_label')

    return label_loop

def bb2symbols(path):

    cif_doc = gemmi.cif.read_file(path)
    block = cif_doc[0]
    symbol_loop = block.find_loop('_atom_site_type_symbol')

    return symbol_loop

def bb2coords(path):

	atoms = read(path)
	pos = atoms.get_positions()
	fpos = atoms.get_scaled_positions()

	x = pos[:, 0]
	y = pos[:, 1]
	z = pos[:, 2]

	f_x = fpos[:, 0]
	f_y = fpos[:, 1]
	f_z = fpos[:, 2]

	return (x, y, z), (f_x, f_y, f_z)

def bb2bonds(path):

	cif_doc = gemmi.cif.read_file(path)
	block = cif_doc[0] 
	bond_label1_loop = block.find_loop('_geom_bond_atom_site_label_1')
	bond_label2_loop = block.find_loop('_geom_bond_atom_site_label_2')
	bond_dis_loop = block.find_loop('_geom_bond_distance')
	bond_sym_loop = block.find_loop('_geom_bond_site_symmetry_2')
	bond_type_loop = block.find_loop('_ccdc_geom_bond_type')

	return bond_label1_loop, bond_label2_loop, bond_dis_loop, bond_sym_loop, bond_type_loop

def bb2array(path):

	_, unit_cell = bb2lattices(path)
	_, (x_loop, y_loop, z_loop) = bb2coords(path)
	label_loop = bb2labels(path)

	fcoords = []
	for i in range(len(label_loop)):
		fcoords.append([label_loop[i],[float(x_loop[i]), float(y_loop[i]), float(z_loop[i])]])

	norm_vec = fcoords[0][1]
	ccoords = [[n[0],np.dot(unit_cell, PBC3DF(norm_vec, n[1]))] for n in fcoords]
	
	com = np.average(np.array([n[1] for n in ccoords if re.sub('[0-9]', '', n[0]) == 'X']), axis = 0)
	sccoords = [[n[0], n[1] - com] for n in ccoords]

	return sccoords

def X_vecs(path, label):

	_, unit_cell = bb2lattices(path)
	_, (x_loop, y_loop, z_loop) = bb2coords(path)
	label_loop = bb2labels(path)

	if label:
		fcoords = []
		for i in range(len(label_loop)):
			if label_loop[i][0] == "X":
				fcoords.append([label_loop[i],[float(x_loop[i]), float(y_loop[i]), float(z_loop[i])]])
		
		mic_fcoords = [[vec[0], PBC3DF(fcoords[0][1], vec[1])] for vec in fcoords]

		ccoords = [[vec[0], np.dot(unit_cell,vec[1])] for vec in mic_fcoords]
		com = np.average(np.asarray([vec[1] for vec in ccoords]), axis=0)
		shifted_ccoords = [[vec[0], vec[1] - com] for vec in ccoords]

	else:
		fcoords = []
		for i in range(len(label_loop)):
			if label_loop[i][0] == "X":
				fcoords.append([float(x_loop[i]), float(y_loop[i]), float(z_loop[i])])

		mic_fcoords = [PBC3DF(fcoords[0], vec) for vec in fcoords]

		ccoords = [np.dot(unit_cell, vec) for vec in mic_fcoords]
		com = np.average(ccoords, axis=0)
		shifted_ccoords = [vec - com for vec in ccoords]

	return shifted_ccoords

def calc_edge_len(path):

	_, unit_cell = bb2lattices(path)
	_, (x_loop, y_loop, z_loop) = bb2coords(path)
	label_loop = bb2labels(path)

	fcoords = []
	for i in range(len(label_loop)):
		if label_loop[i][0] == "X":
			fcoords.append([float(x_loop[i]), float(y_loop[i]), float(z_loop[i])])
	
	mic_fcoords = [PBC3DF(fcoords[0], vec) for vec in fcoords]
	ccoords = [np.dot(unit_cell, vec) for vec in mic_fcoords]
	return np.linalg.norm(ccoords[0] - ccoords[1])

def cncalc(path):

	label_loop = bb2labels(path)
	
	cn = 0
	for label in label_loop:
		if label[0] == 'X':
			cn += 1
	return cn