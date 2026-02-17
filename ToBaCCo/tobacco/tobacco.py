from __future__ import print_function

import os
import sys
sys.path.append(os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from ciftemplate2graph import ct2g
from vertex_edge_assign import vertex_assign, assign_node_vecs2edges
from cycle_cocyle import cycle_cocyle, Bstar_alpha
from bbcif_properties import cncalc, bb2symbols
from SBU_geometry import SBU_coords
from scale import scale
from scaled_embedding2coords import omega2coords
from place_bbs import scaled_node_and_edge_vectors, place_nodes, place_edges
from remove_dummy_atoms import remove_Fr
from adjust_edges import adjust_edges
from write_cifs import write_cif, bond_connected_components, distance_search_bond, fix_bond_sym, merge_catenated_cifs
from ase.geometry import cellpar_to_cell
from utils import vname_dict, metal_elements

import os
import re
import numpy as np
import itertools
import time


def run_tobacco(template, nodes, edges, save_path,
				connection_site_bond_length=1.54,
				symmetry_tol={2:0.10, 3:0.12, 4:0.35, 5:0.25, 6:0.45, 7:0.35, 8:0.40, 9:0.60, 10:0.60, 12:0.60},
				all_node_combinations=False, combinatorial_edge_assignment=True,
				n_iters=1, pre_scale=1, bond_tol=5, min_cell_length=5, fix_uc=(0,0,0,0,0,0)):

	start_time = time.time()

	print('template:', template)
	print()
	
	cat_count = 0
	for net in ct2g(template):
		cat_count += 1
		TG, start, unit_cell, TVT, TET, TNAME, a, b, c, ang_alpha, ang_beta, ang_gamma, max_le, catenation = net

		TVT = sorted(TVT,
					key=lambda x:x[0],
					reverse=True)
		TET = sorted(TET,
			   		reverse=True)

		node_cns = [(cncalc(node), node) for node in nodes]
		
		print('Number of vertices = ', len(TG.nodes()))
		print('Number of edges = ', len(TG.edges()))
		print()

		topo_name = os.path.basename(template)

		edge_counts = dict((data['type'],0) for _, _, data in TG.edges(data=True))
		for _, _, data in TG.edges(data=True):
			edge_counts[data['type']] += 1
	
		vas = vertex_assign(TG,
							TVT,
							node_cns,
							unit_cell,
							symmetry_tol,
							all_node_combinations
					  		)
		CB, CO = cycle_cocyle(TG)

		for va in vas:
			if len(va) == 0:
				print('At least one vertex does not have a building block with the correct number of connection sites.')
				print('Moving to the next template...')
				print()
				continue
	
		if len(CB) != (len(TG.edges()) - len(TG.nodes()) + 1):
			print('The cycle basis is incorrect.')
			print('The number of cycles in the cycle basis does not equal the rank of the cycle space.')
			print('Moving to the next tempate...')
			continue
		
		num_edges = len(TG.edges())
		Bstar, alpha = Bstar_alpha(CB,CO,TG,num_edges)
	
		num_vertices = len(TG.nodes())
	
		if combinatorial_edge_assignment:
			eas = list(itertools.product(sorted(edges), repeat=len(TET)))
		else:
			edge_files = sorted(edges)
			ea = []
			i = 0
			while len(ea) < len(TET):
				ea.append(edge_files[i])
				i += 1
				if i == len(edge_files):
					i = 0
			eas = [tuple(ea)]
	
		g = 0

		for va in vas:
	
			node_elems = [bb2symbols(i[1]) for i in va]
			metals = [[i for i in j if i in metal_elements] for j in node_elems]
			metals = list(set([i for j in metals for i in j]))
	
			v_set = [('v' + str(vname_dict[re.sub('[0-9]','',i[0])]), i[1]) for i in va]
			v_set = sorted(list(set(v_set)), key=lambda x: x[0])
			v_set = [v[0] + '-' + v[1] for v in v_set]
	
			print('vertex assignment:', v_set)

			for v in va:
				for n in TG.nodes(data=True):
					if v[0] == n[0]:
						n[1]['cifname'] = v[1]
	
			for ea in eas:
	
				g += 1
	
				print('edge assignment:', ea)
				print()

				type_assign = dict((k,[]) for k in sorted(TET, reverse=True))
			
				for k,m in zip(TET, ea):
					type_assign[k] = m

				for e in TG.edges(data=True):
					ty = e[2]['type']
					for k in type_assign:
						if ty == k or (ty[1], ty[0]) == k:
							e[2]['cifname'] = type_assign[k]

				num_possible_XX_bonds = 0
				for edge_type, cifname in zip(TET, ea):
					if cifname == 'ntn_edge.cif':
						factor = 1
					else:
						factor = 2
					edge_type_count = edge_counts[edge_type]
					num_possible_XX_bonds += factor * edge_type_count

				ea_dict = assign_node_vecs2edges(TG, unit_cell, template)
		
				all_SBU_coords = SBU_coords(TG, ea_dict, connection_site_bond_length)
				sc_a, sc_b, sc_c, sc_alpha, sc_beta, sc_gamma, sc_covar, Bstar_inv, max_length, _, _, _, _ = scale(all_SBU_coords,
																									   a,b,c,
																									   ang_alpha,
																									   ang_beta,
																									   ang_gamma,
																									   max_le,
																									   num_vertices,
																									   Bstar,
																									   alpha,
																									   num_edges,
																									   fix_uc,
																									   n_iters,
																									   pre_scale,
																									   min_cell_length)
		
				print('The scaled unit cell parameters')
				print('a    :', np.round(sc_a, 5))
				print('b    :', np.round(sc_b, 5))
				print('c    :', np.round(sc_c, 5))
				print('alpha:', np.round(sc_alpha, 5))
				print('beta :', np.round(sc_beta, 5))
				print('gamma:', np.round(sc_gamma, 5))
				print()
	
				for sc, name in zip((sc_a, sc_b, sc_c), ('a', 'b', 'c')):
					cflag = False
					if sc == min_cell_length:
						print('unit cell parameter', name, 'may have collapsed during scaling!')
						print('try re-running with', name, 'fixed or a larger min_cell_length')
						print('no cif will be written')
						cflag = True
	
				if cflag:
					continue
	
				scaled_params = [sc_a,sc_b,sc_c,sc_alpha,sc_beta,sc_gamma]
			
				sc_Alpha = np.r_[alpha[0:num_edges-num_vertices+1,:], sc_covar]
				sc_omega_plus = np.dot(Bstar_inv, sc_Alpha)
			
				cell = cellpar_to_cell([sc_a, sc_b, sc_c, sc_alpha, sc_beta, sc_gamma])
				sc_unit_cell = np.asarray(cell).T
				
				scaled_coords = omega2coords(start, TG, sc_omega_plus, num_vertices)
				nvecs,evecs = scaled_node_and_edge_vectors(scaled_coords, sc_omega_plus, sc_unit_cell, ea_dict)
				placed_nodes, node_bonds = place_nodes(nvecs)
				placed_edges, edge_bonds = place_edges(evecs, len(placed_nodes))
				placed_edges = adjust_edges(placed_edges, placed_nodes, sc_unit_cell)
				placed_nodes = np.c_[placed_nodes, np.array(['node' for i in range(len(placed_nodes))])]
				placed_edges = np.c_[placed_edges, np.array(['edge' for i in range(len(placed_edges))])]

				placed_all = list(placed_nodes) + list(placed_edges)
				bonds_all = node_bonds + edge_bonds
		
				placed_all, bonds_all, nconnections = remove_Fr(placed_all,bonds_all)
				
				print('computing X-X bonds...')
				print()
				print('Bond formation')
				
				fixed_bonds, nbcount, bond_check_passed = bond_connected_components(placed_all, bonds_all, sc_unit_cell, max_length, bond_tol, nconnections, num_possible_XX_bonds)
				print('there were ', nbcount, ' X-X bonds formed')
					
				if bond_check_passed:
					print('bond check passed')
					bond_check_code = ''
				else:
					print('bond check failed, attempting distance search bonding...')
					fixed_bonds, nbcount = distance_search_bond(placed_all, bonds_all, sc_unit_cell, 2.5)
					bond_check_code = '_BOND_CHECK_FAILED'
					print('there were', nbcount, 'X-X bonds formed')
				print()
		
				fc_placed_all = placed_all
	
				fixed_bonds = fix_bond_sym(fixed_bonds, placed_all, sc_unit_cell)

				vnames = '_'.join([os.path.splitext(os.path.basename(v))[0] for v in v_set])
				enames_list = [os.path.splitext(os.path.basename(e))[0] for e in ea]
				enames_grouped = [list(edge_gr) for ind,edge_gr in itertools.groupby(enames_list)]
				enames_grouped = [(len(edge_gr), list(set(edge_gr))) for edge_gr in enames_grouped]
				enames_flat = [str(L) + '-' + '_'.join(names) for L,names in enames_grouped]
				enames = '_'.join(enames_flat)
				cat_cifs = []
				if catenation:
					cifname = topo_name[0:-4] + '_' +  vnames + '_' + enames + bond_check_code + '_' + 'CAT' + str(cat_count) + '.cif'
					save_name = os.path.join(save_path, cifname)
					cat_cifs.append(save_name)
				else:
					cifname = topo_name[0:-4] + '_' +  vnames + '_' + enames + bond_check_code + '.cif'
					save_name = os.path.join(save_path, cifname)

				print('writing cif:', save_name)
				print()
				
				write_cif(fc_placed_all, fixed_bonds, scaled_params, sc_unit_cell, save_name)

	if catenation:
		
		print('merging catenated cifs...')

		for comb in itertools.combinations(cat_cifs, cat_count):
			builds = [name[0:-9] for name in comb]
			print(set(builds))
			if len(set(builds)) == 1:
				pass
			else:
				continue
			merge_catenated_cifs(comb)

	print('Run time: %0f seconds' % (time.time() - start_time))
	print('-'*30)
	print()
