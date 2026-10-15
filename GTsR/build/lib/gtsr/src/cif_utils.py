from __future__ import annotations

import os, collections, stat, shutil, subprocess, sys, tempfile
from pathlib import Path
from pymatgen.core import Structure
from ase.io import read
from ase.build import sort
from ase import neighborlist
from pymatgen.io.ase import AseAtomsAdaptor
from ase.data import atomic_numbers
import numpy as np
from pymatgen.io.cif import CifWriter
import networkx as nx
from ase import Atoms

from molSimplify.Informatics.MOF.MOF_descriptors import get_MOF_descriptors

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


def convert2pymatgen(cif_path: str | Path):
    struc = Structure.from_file(cif_path)
    CifWriter(struc).write_file(cif_path)


def cif2graph(
                cif_path: str | Path,
                radius: float = 8.0,
                max_neighbors: int = 12,
            ) -> dict:
    
    structure = read(cif_path)
    struct = AseAtomsAdaptor.get_structure(structure)

    ele_list = [sp.symbol for sp in struct.species]
    numbers = [atomic_numbers[sym] for sym in ele_list]
    c_index, n_index, _, dists = struct.get_neighbor_list(
        r=radius, numerical_tol=0, exclude_self=True)
    n_sites = len(struct)
    selected_idx_chunks = []
    for i in range(n_sites):
        mask = (c_index == i)
        if np.any(mask):
            idx = np.where(mask)[0]
            order = np.argsort(dists[idx])[:max_neighbors]
            selected_idx_chunks.append(idx[order])
    if selected_idx_chunks:
        selected_idx = np.concatenate(selected_idx_chunks)
        index1 = c_index[selected_idx]
        index2 = n_index[selected_idx]
        dij    = dists[selected_idx]
    else:
        index1 = np.array([], dtype=int)
        index2 = np.array([], dtype=int)
        dij    = np.array([], dtype=float)

    return {
            "rcut": float(radius),
            "max_neighbors": int(max_neighbors),
            "numbers": numbers,
            "index1": index1.tolist(),
            "index2": index2.tolist(),
            "dij": dij.tolist()
            }
        
def cif2pos(cif_path: str | Path) -> np.ndarray:
    struct = Structure.from_file(cif_path)
    return struct.frac_coords


def make_label(cif_path1: str | Path, cif_path2: str | Path, tol=1e-3) -> np.ndarray:

    struct1 = Structure.from_file(cif_path1)
    struct2 = Structure.from_file(cif_path2)

    dists = struct1.lattice.get_all_distances(struct1.frac_coords, struct2.frac_coords)
    min_dists = np.min(dists, axis=1)
    labels = np.where(min_dists < tol, 0, 1)
     
    return labels
    
def label2cif(cif_path: str | Path, label: np.ndarray, output_dir: str) -> Structure:
    struct = Structure.from_file(cif_path)
    
    solvent = label == 1
    framework = ~solvent

    def extract(mask):
        indices = np.where(mask)[0]
        sites   = [struct[i] for i in indices]
        return Structure.from_sites(sites)
    
    os.makedirs(output_dir, exist_ok=True)

    framework_atoms = extract(framework)
    stem = Path(cif_path).stem
    CifWriter(framework_atoms).write_file(os.path.join(output_dir, stem + "_gtsr.cif"))
    if solvent.any():
        CifWriter(extract(solvent)).write_file(os.path.join(output_dir, stem + "_sol.cif"))
        return extract(solvent)

    return None


def dict2str(dct):
    return ''.join(symb + (str(n)) for symb, n in dct.items())

def split_from_cif(cif_path):

    atoms = read(cif_path)

    cutOff = neighborlist.natural_cutoffs(atoms)
    neighborList = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True, skin=0.3)
    neighborList.update(atoms)
    G = nx.Graph()
    for k in range(len(atoms)):
        tup = (k, {"element":"{}".format(atoms.get_chemical_symbols()[k]), "pos": atoms.get_positions()[k]})
        G.add_nodes_from([tup])
    for k in range(len(atoms)):
        for i in neighborList.get_neighbors(k)[0]:
            G.add_edge(k, i)
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    form_dicts = []
    for index, g in enumerate(Gcc):
        g = list(g)
        fragment = atoms[g]
        fragment = sort(fragment)
        form_dict = fragment.symbols.formula.count()
        form_dicts.append(dict2str(form_dict))
    nodes = []
    unique_formdicts = []
    if len(form_dicts) > 1:
        for index, form_dict in enumerate(form_dicts):
            if form_dict not in unique_formdicts:
                nodes.append(atoms[list(Gcc[index])])
                unique_formdicts.append(form_dict)
    elif len(form_dicts) == 1:
        nodes.append(atoms[list(Gcc[0])])
        unique_formdicts.append(form_dicts[0])
    xyzs = []

    for index, _ in enumerate(nodes):
        xyzs.append(remove_pbc_cuts(nodes[index]))
    return len(nodes), xyzs


def remove_pbc_cuts(atoms):
    try:
        scale = 1.4
        cutoffs = neighborlist.natural_cutoffs(atoms)
        cutoffs = [scale * c for c in cutoffs]
        I, J, D = neighborlist.neighbor_list("ijD",atoms,cutoff=cutoffs)
        nl = [[] for _ in atoms]
        for i, j, d in zip(I, J, D):
            nl[i].append((j, d))
        visited = [False for _ in atoms]
        q = collections.deque()
        abc_half = np.sum(atoms.get_cell(), axis=0) * 0.5
        positions = {}
        q.append((0, np.array([0.0, 0.0, 0.0])))
        while q:
            i, pos = q.pop()
            visited[i] = True
            positions[i] = pos
            for j, d in nl[i]:
                if not visited[j]:
                    q.append((j, pos + d))
                    visited[j] = True
        centroid = np.array([0.0, 0.0, 0.0])
        for v in positions.values():
            centroid += v
        centroid /= len(positions)
        syms = [None for _ in atoms]
        poss = [None for _ in atoms]
        for i in range(len(atoms)):
            syms[i] = atoms.symbols[i]
            poss[i] = positions[i] - centroid + abc_half
        atoms = Atoms(
            symbols=syms, positions=poss, pbc=True, cell=atoms.get_cell()
        )
        cell_x = np.max(atoms.positions[:,0]) - np.min(atoms.positions[:,0])
        cell_y = np.max(atoms.positions[:,1]) - np.min(atoms.positions[:,1])
        cell_z = np.max(atoms.positions[:,2]) - np.min(atoms.positions[:,2])
        cell = max([cell_x,cell_y,cell_z])
        atoms.set_cell([cell+2,cell+2,cell+2, 90,90,90])
        center_mass = atoms.get_center_of_mass()
        cell_half  = atoms.cell.cellpar()[0:3]/2
        atoms.positions = atoms.positions - center_mass + cell_half    
        return atoms
    except:
        return atoms
    

def xyz_to_smiles(xyz_data: str, charge: int = 0) -> str:

    raw_mol = Chem.MolFromXYZBlock(xyz_data)

    conn_mol = Chem.RWMol(raw_mol)
    rdDetermineBonds.DetermineConnectivity(conn_mol)

    for try_charge in [charge, -1, 1, -2, 2, 0]:
        try:
            mol_copy = Chem.RWMol(conn_mol)
            rdDetermineBonds.DetermineBondOrders(mol_copy, charge=try_charge)
            Chem.SanitizeMol(mol_copy)
            return Chem.MolToSmiles(mol_copy)
        except Exception:
            continue
    Chem.SanitizeMol(conn_mol, catchErrors=True)
    return Chem.MolToSmiles(conn_mol)


def canonical_smiles(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    return Chem.MolToSmiles(mol)


def atoms2xyzblock(atoms) -> str:
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    lines = [
        str(len(symbols)),
        "generated from ASE Atoms",
    ]

    for symbol, pos in zip(symbols, positions):
        x, y, z = pos
        lines.append(f"{symbol} {x:.8f} {y:.8f} {z:.8f}")

    return "\n".join(lines) + "\n"

def get_sol_smi(cif_path):

    smis = []

    _, xyzs = split_from_cif(cif_path)
    for i, sol in enumerate(xyzs):
        smi = xyz_to_smiles(atoms2xyzblock(sol))
        can_smi = canonical_smiles(smi)
        if can_smi not in smis:
            smis.append(can_smi)
        else:
            pass
    return smis


def remove_dir_with_permissions(dir_path):
    def handle_permission_error(func, path, exc_info):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, onerror=handle_permission_error)


def RACs(cif_path):
    tmp_dir = Path(tempfile.mkdtemp(prefix="gtsr-rac-"))
    try:
        name = Path(cif_path).stem
        full_names, full_descriptors = get_MOF_descriptors(
            cif_path,
            3,
            path=str(tmp_dir),
            xyz_path=str(tmp_dir / f"{name}.xyz"),
            max_num_atoms=6000,
        )
        return dict(zip(full_names, full_descriptors))
    finally:
        remove_dir_with_permissions(tmp_dir)


def get_cell(cif_path):
    structure = Structure.from_file(cif_path)
    return structure.lattice.matrix


def flatten_rac(descriptor_data):
    return {f"{k}": v for k, v in descriptor_data.items()}


def flatten_cell(cell):
    flat = {}
    for r in range(3):
        for c in range(3):
            flat[f"cell_{r}{c}"] = cell[r, c]
    return flat


def n_atom(cif_path):
    return len(read(cif_path))


def _network_executable():
    executable = shutil.which("network")
    if executable:
        return executable

    environment_executable = Path(sys.executable).with_name("network")
    if environment_executable.is_file():
        return str(environment_executable)

    raise FileNotFoundError(
        "Zeo++ executable 'network' was not found in PATH or beside the "
        f"current Python interpreter: {sys.executable}"
    )


def PoreDiameter(cif_path):
    tmp_dir = Path(tempfile.mkdtemp(prefix="gtsr-pd-"))
    tmp_file = tmp_dir / f"pd.txt"
    try:
        subprocess.run(
            [_network_executable(), "-ha", "-res", str(tmp_file), str(cif_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        with tmp_file.open() as file:
            line = file.readline().split()
            return dict(zip(("Di", "Df", "Dif"), map(float, line[1:4])))
    finally:
        remove_dir_with_permissions(tmp_dir)


def PoreVolume(cif_path):
    tmp_dir = Path(tempfile.mkdtemp(prefix="gtsr-pv-"))
    tmp_file = tmp_dir / "pv.txt"
    try:
        subprocess.run(
            [
                _network_executable(),
                "-ha",
                "-volpo",
                "0",
                "0",
                "5000",
                str(tmp_file),
                str(cif_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        with tmp_file.open() as file:
            row = file.readline()
            density = float(row.split("Density:")[1].split()[0])
            poav = float(row.split("POAV_A^3:")[1].split()[0])
            ponav = float(row.split("PONAV_A^3:")[1].split()[0])
            gpoav = float(row.split("POAV_cm^3/g:")[1].split()[0])
            gponav = float(row.split("PONAV_cm^3/g:")[1].split()[0])
            poav_volume_fraction = float(
                row.split("POAV_Volume_fraction:")[1].split()[0]
            )
            ponav_volume_fraction = float(
                row.split("PONAV_Volume_fraction:")[1].split()[0]
            )
        return {
            "PV": [poav, gpoav],
            "NPV": [ponav, gponav],
            "VF": poav_volume_fraction,
            "NVF": ponav_volume_fraction,
            "Density": density,
        }
    finally:
        remove_dir_with_permissions(tmp_dir)
