from __future__ import annotations

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, atom_fea_len: int, nbr_fea_len: int):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.norm_v = nn.LayerNorm(atom_fea_len)
        self.phi_e = nn.Sequential(
            nn.Linear(2 * atom_fea_len + nbr_fea_len, atom_fea_len),
            nn.LeakyReLU(0.2),
            nn.Linear(atom_fea_len, atom_fea_len),
            nn.LeakyReLU(0.2),
            nn.Linear(atom_fea_len, atom_fea_len),
        )
        self.phi_v = nn.Sequential(
            nn.Linear(2 * atom_fea_len, atom_fea_len),
            nn.LeakyReLU(0.2),
            nn.Linear(atom_fea_len, atom_fea_len),
            nn.LeakyReLU(0.2),
            nn.Linear(atom_fea_len, atom_fea_len),
        )

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx1, nbr_fea_idx2, crystal_atom_idx):
        n_atom, width = atom_in_fea.shape
        device = atom_in_fea.device

        if nbr_fea_idx1.numel() == 0:
            rho_e_v = torch.zeros((n_atom, width), device=device, dtype=atom_in_fea.dtype)
            ek_sum = torch.zeros((n_atom, width), device=device, dtype=atom_in_fea.dtype)
            ek = nbr_fea
        else:
            atom_nbr_fea1 = atom_in_fea[nbr_fea_idx1, :]
            atom_nbr_fea2 = atom_in_fea[nbr_fea_idx2, :]
            neighbor_counts = torch.zeros((n_atom,), device=device, dtype=atom_in_fea.dtype)
            neighbor_counts = neighbor_counts.scatter_add(
                0,
                nbr_fea_idx1,
                torch.ones(nbr_fea_idx1.shape, device=device, dtype=atom_in_fea.dtype),
            )
            neighbor_count_fea = neighbor_counts[nbr_fea_idx1].clamp(min=1.0).view(-1, 1)
            total_nbr_fea = torch.cat([atom_nbr_fea1, atom_nbr_fea2, nbr_fea], dim=1)
            ek = self.phi_e(total_nbr_fea)
            rho_e_v = torch.zeros((n_atom, width), device=device, dtype=atom_in_fea.dtype)
            rho_e_v = rho_e_v.scatter_add(
                0,
                nbr_fea_idx1.view(-1, 1).repeat(1, width),
                ek / neighbor_count_fea,
            )
            ek = nbr_fea + ek
            ek_sum = torch.zeros((n_atom, width), device=device, dtype=atom_in_fea.dtype)
            ek_sum = ek_sum.scatter_add(
                0,
                nbr_fea_idx1.view(-1, 1).repeat(1, width),
                ek / neighbor_count_fea,
            )

        total_node_fea = torch.cat([atom_in_fea, rho_e_v], dim=1)
        vi = atom_in_fea + self.norm_v(self.phi_v(total_node_fea))
        atom_nbr_fea = torch.cat([vi, ek_sum], dim=1)

        n_crys = int(crystal_atom_idx.max().item()) + 1 if crystal_atom_idx.numel() else 0
        global_fea = torch.zeros((n_crys, 2 * width), device=device, dtype=atom_in_fea.dtype)
        global_fea = global_fea.scatter_add(
            0,
            crystal_atom_idx.view(-1, 1).repeat(1, 2 * width),
            atom_nbr_fea,
        )
        return ek, vi, global_fea, atom_nbr_fea


class SolventAtomClassifier(nn.Module):
    def __init__(
        self,
        orig_atom_fea_len: int,
        nbr_fea_len: int,
        atom_fea_len: int = 128,
        n_conv: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.edge_embedding = nn.Linear(nbr_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [
                ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=atom_fea_len)
                for _ in range(n_conv)
            ]
        )
        self.local_embedding = nn.Sequential(
            nn.Linear(2 * atom_fea_len, 128),
            nn.LeakyReLU(0.2),
        )
        self.global_embedding = nn.Sequential(
            nn.Linear(2 * atom_fea_len, 128),
            nn.LeakyReLU(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx1, nbr_fea_idx2, atom_idx):
        atom_fea = self.node_embedding(atom_fea)
        nbr_fea = self.edge_embedding(nbr_fea)

        atom_nbr_fea = torch.cat([atom_fea, torch.zeros_like(atom_fea)], dim=1)
        n_crys = int(atom_idx.max().item()) + 1 if atom_idx.numel() else 0
        global_fea = torch.zeros(
            (n_crys, atom_nbr_fea.shape[1]),
            device=atom_fea.device,
            dtype=atom_fea.dtype,
        )

        for conv_func in self.convs:
            nbr_fea, atom_fea, global_fea, atom_nbr_fea = conv_func(
                atom_fea,
                nbr_fea,
                nbr_fea_idx1,
                nbr_fea_idx2,
                atom_idx,
            )

        atom_counts = torch.zeros((n_crys, 1), device=atom_fea.device, dtype=atom_fea.dtype)
        atom_counts = atom_counts.scatter_add(
            0,
            atom_idx.view(-1, 1),
            torch.ones((atom_idx.numel(), 1), device=atom_fea.device, dtype=atom_fea.dtype),
        )
        global_fea = global_fea / atom_counts.clamp(min=1.0)

        local_fea = self.local_embedding(atom_nbr_fea)
        global_atom_fea = self.global_embedding(global_fea[atom_idx])
        logits = self.classifier(torch.cat([local_fea, global_atom_fea], dim=-1)).squeeze(-1)
        return logits
