from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .src.GCN import SolventAtomClassifier
    from .src.cif_utils import (
        PoreDiameter,
        PoreVolume,
        RACs,
        cif2graph,
        cif2pos,
        flatten_cell,
        flatten_rac,
        get_cell,
        get_sol_smi,
        label2cif,
        n_atom,
        convert2pymatgen
    )
    from .src.data import GaussianDistance
    from .src.utils import load_checkpoint
except ImportError:
    from src.GCN import SolventAtomClassifier
    from src.cif_utils import (
        PoreDiameter,
        PoreVolume,
        RACs,
        cif2graph,
        cif2pos,
        flatten_cell,
        flatten_rac,
        get_cell,
        get_sol_smi,
        label2cif,
        n_atom,
        convert2pymatgen
    )
    from src.data import GaussianDistance
    from src.utils import load_checkpoint


PACKAGE_DIR = Path(__file__).resolve().parent


def _bundled_model(filename: str) -> Path:
    candidates = (
        PACKAGE_DIR / "ckpt" / filename,
        PACKAGE_DIR.parent / "ckpt" / filename,
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _bundled_checkpoint(name: str) -> Path:
    return _bundled_model(f"{name}_best.pth")


CHECKPOINTS = {
    "free": _bundled_checkpoint("free"),
    "all": _bundled_checkpoint("all"),
}
DEFAULT_CHECKPOINT = CHECKPOINTS["free"]
STABILITY_MODEL = _bundled_model("stability_best.pkl")
RAC_FEATURE_NAMES = tuple(
    f"{prefix}-{property_name}-{depth}"
    for prefix, property_names in (
        ("f-sbu", ("chi", "Z", "I", "T", "S")),
        ("mc", ("chi", "Z", "I", "T", "S")),
        ("D_mc", ("chi", "Z", "I", "T", "S")),
        ("f-link", ("chi", "Z", "I", "T", "S")),
        ("lc", ("chi", "Z", "I", "T", "S", "alpha")),
        ("D_lc", ("chi", "Z", "I", "T", "S", "alpha")),
        ("func", ("chi", "Z", "I", "T", "S", "alpha")),
        ("D_func", ("chi", "Z", "I", "T", "S", "alpha")),
    )
    for property_name in property_names
    for depth in range(4)
)


class GTsRunner:

    def __init__(
        self,
        checkpoint: str | Path = "",
        device: str | torch.device | None = None,
    ) -> None:
        checkpoint_name = str(checkpoint).strip().lower()
        self.device = self._resolve_device(device)
        self.stability_model = None
        self.stability_imputer = None

        if checkpoint_name == "stability":
            self.checkpoint_path = self._resolve_stability_model()
            self._load_stability_model()
            self.checkpoint = None
            self.task = "stability"
            return

        self.checkpoint_path = self._resolve_checkpoint(checkpoint)
        self.checkpoint = load_checkpoint(self.checkpoint_path, device=self.device)

        model_config = self.checkpoint.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError(
                f"Checkpoint does not contain a valid model_config: {self.checkpoint_path}"
            )

        self.model = SolventAtomClassifier(**model_config).to(self.device)
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.eval()

        self.radius = float(self.checkpoint.get("radius", 8.0))
        self.dmin = float(self.checkpoint.get("dmin", 0.0))
        self.step = float(self.checkpoint.get("step", 0.2))
        self.default_threshold = float(self.checkpoint.get("threshold", 0.5))
        self.task = str(self.checkpoint.get("task", "unknown"))
        self.max_atomic_number = 118
        self.gdf = GaussianDistance(
            dmin=self.dmin,
            dmax=self.radius,
            step=self.step,
        )

    def clean(
        self,
        cif: str | Path = "",
        output: str | Path = "",
        threshold: float | None = None,
    ) -> dict[str, Any]:
        
        convert2pymatgen(cif)

        if self.task == "stability":
            raise RuntimeError(
                "clean() requires a GNN checkpoint; initialize GTsRunner with "
                "checkpoint='free' or checkpoint='all'"
            )

        cif_path = self._resolve_cif(cif)
        output_dir = self._resolve_output(cif_path, output)
        cutoff = self.default_threshold if threshold is None else float(threshold)
        if not 0.0 <= cutoff <= 1.0:
            raise ValueError(f"threshold must be between 0 and 1, got {cutoff}")

        tensors = self._build_tensors(cif_path)
        with torch.inference_mode():
            probabilities = torch.sigmoid(self.model(*tensors)).cpu().numpy()

        labels = (probabilities >= cutoff).astype(np.int64)
        label2cif(cif_path, labels, str(output_dir))

        stem = cif_path.stem
        framework_path = output_dir / f"{stem}_gtsr.cif"
        solvent_path = output_dir / f"{stem}_sol.cif"
        try:
            sol_smis = get_sol_smi(solvent_path)
        except:
            sol_smis = None
        return {
            "input": str(cif_path),
            "output": str(output_dir),
            "framework": str(framework_path),
            "solvent": str(solvent_path) if solvent_path.exists() else None,
            "checkpoint": str(self.checkpoint_path),
            "task": self.task,
            "threshold": cutoff,
            "num_atoms": int(labels.size),
            "num_framework_atoms": int((labels == 0).sum()),
            "num_solvent_atoms": int((labels == 1).sum()),
            "probabilities": probabilities.tolist(),
            "labels": labels.tolist(),
            "solvent_smiles": sol_smis
        }

    def _build_tensors(self, cif_path: Path) -> tuple[torch.Tensor, ...]:
        graph = cif2graph(cif_path, radius=self.radius)
        positions = np.asarray(cif2pos(cif_path), dtype=np.float32)
        numbers = np.asarray(graph["numbers"], dtype=np.int64)

        if numbers.size == 0:
            raise ValueError(f"CIF contains no atoms: {cif_path}")
        if numbers.min() < 1 or numbers.max() > self.max_atomic_number:
            raise ValueError(
                f"CIF contains an unsupported atomic number; supported range is "
                f"1-{self.max_atomic_number}"
            )
        if len(positions) != len(numbers):
            raise ValueError(
                f"Position/atom mismatch in {cif_path}: "
                f"{len(positions)} positions for {len(numbers)} atoms"
            )

        atom_features = np.eye(
            self.max_atomic_number + 1,
            dtype=np.float32,
        )[numbers]
        atom_features = np.concatenate([atom_features, positions], axis=1)

        distances = np.asarray(graph["dij"], dtype=np.float32)
        neighbor_features = self.gdf.expand(distances).astype(np.float32)
        index1 = np.asarray(graph["index1"], dtype=np.int64)
        index2 = np.asarray(graph["index2"], dtype=np.int64)
        atom_index = np.zeros(len(numbers), dtype=np.int64)

        tensors = (
            torch.from_numpy(atom_features),
            torch.from_numpy(neighbor_features),
            torch.from_numpy(index1),
            torch.from_numpy(index2),
            torch.from_numpy(atom_index),
        )
        return tuple(tensor.to(self.device) for tensor in tensors)

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is None or str(device).lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return resolved

    @staticmethod
    def _resolve_checkpoint(checkpoint: str | Path) -> Path:
        checkpoint_name = str(checkpoint).strip().lower()
        if not checkpoint_name:
            path = DEFAULT_CHECKPOINT
        elif checkpoint_name in CHECKPOINTS:
            path = CHECKPOINTS[checkpoint_name]
        else:
            path = Path(checkpoint).expanduser()
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    @staticmethod
    def _resolve_stability_model() -> Path:
        path = STABILITY_MODEL.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Stability model not found: {path}")
        return path

    def _load_stability_model(self) -> None:
        model_path = self._resolve_stability_model()
        with model_path.open("rb") as model_file:
            saved_model = pickle.load(model_file)

        if isinstance(saved_model, dict):
            self.stability_model = saved_model["model"]
            self.stability_imputer = saved_model.get("imputer")
        else:
            self.stability_model = saved_model
            self.stability_imputer = None

        self._make_stability_model_compatible()

    def _make_stability_model_compatible(self) -> None:
        """Fill attributes absent from models saved by older scikit-learn versions."""
        estimators = [self.stability_model]
        estimators.extend(getattr(self.stability_model, "estimators_", []))
        for estimator in estimators:
            if estimator is not None and not hasattr(estimator, "monotonic_cst"):
                estimator.monotonic_cst = None

    @staticmethod
    def _resolve_cif(cif: str | Path) -> Path:
        if not cif:
            raise ValueError("cif must be a path to an input CIF file")
        path = Path(cif).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CIF not found: {path}")
        return path

    @staticmethod
    def _resolve_output(cif_path: Path, output: str | Path) -> Path:
        path = (
            Path(output).expanduser()
            if output
            else cif_path.parent / f"{cif_path.stem}_gtsr"
        )
        path = path.resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def stability(self, cif: str | Path):
        cif_path = self._resolve_cif(cif)
        cif_filename = str(cif_path)
        cell = flatten_cell(get_cell(cif_filename))
        pore_diameter = PoreDiameter(cif_filename)
        pore_volume = PoreVolume(cif_filename)
        rac = flatten_rac(RACs(cif_filename))

        features = [
            n_atom(cif_filename),
            *cell.values(),
            pore_diameter["Di"],
            pore_diameter["Df"],
            pore_diameter["Dif"],
            pore_volume["Density"],
            pore_volume["VF"],
            *(rac.get(name, np.nan) for name in RAC_FEATURE_NAMES),
        ]
        feature_batch = np.asarray([features], dtype=np.float64)

        if self.stability_model is None:
            self._load_stability_model()
        if self.stability_imputer is not None:
            feature_batch = self.stability_imputer.transform(feature_batch)

        return self.stability_model.predict(feature_batch)[0]
