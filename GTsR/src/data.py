from __future__ import annotations

import csv
import functools
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate


def get_data_loader(
    dataset,
    collate_fn=default_collate,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    test: bool = False,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not test,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


def collate_pool(dataset_list):
    batch_atom_fea = []
    batch_nbr_fea = []
    batch_nbr_fea_idx1 = []
    batch_nbr_fea_idx2 = []
    crystal_atom_idx = []
    batch_target = []
    batch_pos = []
    batch_dij = []
    batch_cif_ids = []
    base_idx = 0

    for i, ((atom_fea, nbr_fea, nbr_fea_idx1, nbr_fea_idx2, dij), pos, target, cif_id) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_dij.append(dij)
        batch_nbr_fea_idx1.append(nbr_fea_idx1 + base_idx)
        batch_nbr_fea_idx2.append(nbr_fea_idx2 + base_idx)
        crystal_atom_idx.append(torch.LongTensor([i] * n_i))
        batch_target.append(target)
        batch_pos.append(pos)
        batch_cif_ids.append(cif_id)
        base_idx += n_i

    return (
        torch.cat(batch_atom_fea, dim=0),
        torch.cat(batch_nbr_fea, dim=0),
        torch.cat(batch_nbr_fea_idx1, dim=0),
        torch.cat(batch_nbr_fea_idx2, dim=0),
        torch.cat(crystal_atom_idx, dim=0),
        torch.cat(batch_dij, dim=0),
        torch.cat(batch_pos, dim=0),
    ), torch.cat(batch_target, dim=0), batch_cif_ids


class GaussianDistance:
    def __init__(self, dmin: float, dmax: float, step: float, var: float | None = None):
        if dmin >= dmax:
            raise ValueError("dmin must be smaller than dmax")
        if dmax - dmin <= step:
            raise ValueError("dmax - dmin must be larger than step")
        self.filter = np.arange(dmin, dmax + step, step)
        self.var = step if var is None else var

    def expand(self, distances):
        return np.exp(-((distances[..., np.newaxis] - self.filter) ** 2) / self.var**2)


class SolventCIFData(Dataset):
    def __init__(
        self,
        graph_dir: str | Path,
        pos_dir: str | Path,
        label_dir: str | Path,
        csv_file: str | Path,
        radius: float = 8.0,
        dmin: float = 0.0,
        step: float = 0.2,
        max_atomic_number: int = 118,
    ):
        self.graph_dir = Path(graph_dir)
        self.pos_dir = Path(pos_dir)
        self.label_dir = Path(label_dir)
        self.radius = radius
        self.max_atomic_number = max_atomic_number

        with Path(csv_file).open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.reader(handle) if row]
        if rows and rows[0][0].lower() in {"name", "cif_id", "folder_name"}:
            rows = rows[1:]
        self.id_prop_data = [row[0] for row in rows]
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    def _one_hot(self, atomic_number: int) -> np.ndarray:
        fea = np.zeros((self.max_atomic_number + 1,), dtype=np.float32)
        if atomic_number < 1 or atomic_number > self.max_atomic_number:
            raise ValueError(
                f"Atomic number {atomic_number} exceeds max_atomic_number={self.max_atomic_number}"
            )
        fea[atomic_number] = 1.0
        return fea

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        cif_id = self.id_prop_data[idx]
        with (self.graph_dir / f"{cif_id}.json").open(encoding="utf-8") as handle:
            crystal_data = json.load(handle)

        numbers = crystal_data["numbers"]
        atom_fea = np.vstack([self._one_hot(int(number)) for number in numbers])
        pos = np.load(self.pos_dir / f"{cif_id}.npy").astype(np.float32)
        target = np.load(self.label_dir / f"{cif_id}.npy").astype(np.float32)

        if len(pos) != len(numbers):
            raise ValueError(
                f"Position length mismatch for {cif_id}: {len(pos)} positions, {len(numbers)} atoms"
            )
        if len(target) != len(numbers):
            raise ValueError(
                f"Label length mismatch for {cif_id}: {len(target)} labels, {len(numbers)} atoms"
            )

        index1 = np.array(crystal_data["index1"], dtype=np.int64)
        index2 = np.array(crystal_data["index2"], dtype=np.int64)
        dij_np = np.array(crystal_data["dij"], dtype=np.float32)
        nbr_fea = self.gdf.expand(dij_np).astype(np.float32)

        return (
            torch.Tensor(atom_fea),
            torch.Tensor(nbr_fea),
            torch.LongTensor(index1),
            torch.LongTensor(index2),
            torch.Tensor(dij_np),
        ), torch.Tensor(pos), torch.Tensor(target), cif_id
