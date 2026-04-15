"""
Molecular Property Prediction Environment.

Loads molecular datasets (ESOL, FreeSolv, Lipophilicity, hERG, CYP2D6)
and provides four distinct molecular representations:
  - Morgan fingerprints (for Linear agent)
  - RDKit descriptors (for MLP agent)
  - SMILES character encoding (for CNN agent)
  - Atom-level features (for Attention agent)

Each representation captures different aspects of molecular structure,
creating genuine architectural complementarity.
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# Suppress RDKit deprecation warnings (MorganGenerator API change)
warnings.filterwarnings("ignore", message=".*DEPRECATION WARNING.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class MolecularDataset:
    """Container for a molecular dataset with multiple representations."""
    name: str
    task_type: str  # "regression" or "classification"
    # Per-representation feature matrices
    fingerprints: torch.Tensor     # (n_mols, fp_bits) - Morgan fingerprints
    descriptors: torch.Tensor      # (n_mols, n_desc) - RDKit descriptors
    smiles_encoded: torch.Tensor   # (n_mols, max_len) - character-level SMILES
    atom_features: torch.Tensor    # (n_mols, max_atoms * atom_feat_dim) - flattened atom features
    # Targets
    targets: torch.Tensor          # (n_mols, 1)
    # Scaffold split indices
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    # Metadata
    smiles: list[str]
    scaffolds: Optional[np.ndarray] = None


# SMILES character vocabulary
SMILES_CHARS = list("CNOSPFIBrcnos()[]=#-+\\/@.12345678 ")
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(SMILES_CHARS)}  # 0 = padding
VOCAB_SIZE = len(SMILES_CHARS) + 1
MAX_SMILES_LEN = 120


def smiles_to_encoding(smiles: str, max_len: int = MAX_SMILES_LEN) -> np.ndarray:
    """Encode SMILES string as integer sequence."""
    enc = np.zeros(max_len, dtype=np.int64)
    for i, c in enumerate(smiles[:max_len]):
        enc[i] = CHAR_TO_IDX.get(c, 0)
    return enc


def compute_morgan_fingerprint(mol, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """Compute Morgan (ECFP) fingerprint."""
    from rdkit.Chem import AllChem
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def compute_rdkit_descriptors(mol) -> np.ndarray:
    """Compute standard RDKit molecular descriptors."""
    from rdkit.Chem import Descriptors
    desc_list = [
        Descriptors.MolWt,
        Descriptors.MolLogP,
        Descriptors.TPSA,
        Descriptors.NumHDonors,
        Descriptors.NumHAcceptors,
        Descriptors.NumRotatableBonds,
        Descriptors.RingCount,
        Descriptors.NumAromaticRings,
        Descriptors.FractionCSP3,
        Descriptors.HeavyAtomCount,
        Descriptors.NumValenceElectrons,
        Descriptors.NumRadicalElectrons,
        Descriptors.MaxPartialCharge,
        Descriptors.MinPartialCharge,
        Descriptors.MaxAbsPartialCharge,
        Descriptors.MinAbsPartialCharge,
        Descriptors.BertzCT,
        Descriptors.Chi0,
        Descriptors.Chi1,
        Descriptors.HallKierAlpha,
        Descriptors.Kappa1,
        Descriptors.Kappa2,
        Descriptors.LabuteASA,
        Descriptors.BalabanJ,
    ]
    values = []
    for desc_fn in desc_list:
        try:
            v = desc_fn(mol)
            if v is None or not np.isfinite(v):
                v = 0.0
        except Exception:
            v = 0.0
        values.append(float(v))
    return np.array(values, dtype=np.float32)


def compute_atom_features(mol, max_atoms: int = 60) -> np.ndarray:
    """Compute per-atom features, padded/truncated to max_atoms.

    Features per atom (10): atomic_num, degree, formal_charge, num_Hs,
    hybridization, aromatic, in_ring, mass, num_radical_electrons, chirality
    """
    from rdkit.Chem import rdchem
    atom_feat_dim = 10
    features = np.zeros((max_atoms, atom_feat_dim), dtype=np.float32)

    for i, atom in enumerate(mol.GetAtoms()):
        if i >= max_atoms:
            break
        features[i, 0] = atom.GetAtomicNum() / 53.0  # normalize by iodine
        features[i, 1] = atom.GetDegree() / 6.0
        features[i, 2] = atom.GetFormalCharge() / 2.0
        features[i, 3] = atom.GetTotalNumHs() / 4.0
        features[i, 4] = int(atom.GetHybridization()) / 6.0
        features[i, 5] = float(atom.GetIsAromatic())
        features[i, 6] = float(atom.IsInRing())
        features[i, 7] = atom.GetMass() / 127.0  # normalize by iodine mass
        features[i, 8] = atom.GetNumRadicalElectrons() / 2.0
        try:
            features[i, 9] = float(atom.GetChiralTag() != rdchem.ChiralType.CHI_UNSPECIFIED)
        except Exception:
            features[i, 9] = 0.0

    return features.flatten()  # (max_atoms * atom_feat_dim,)


def scaffold_split(smiles_list: list[str], frac_train: float = 0.7,
                   frac_val: float = 0.1, seed: int = 42) -> tuple:
    """Scaffold-based split (Bemis-Murcko scaffolds).

    Molecules with the same scaffold go to the same split.
    This tests generalization to novel chemical scaffolds.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import defaultdict

    scaffold_sets = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffold_sets["INVALID"].append(i)
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )
        except Exception:
            scaffold = smi
        scaffold_sets[scaffold].append(i)

    # Sort scaffolds by size (largest first for determinism)
    scaffold_groups = sorted(scaffold_sets.values(), key=len, reverse=True)

    rng = np.random.RandomState(seed)
    rng.shuffle(scaffold_groups)

    # Flatten all indices in scaffold-grouped order
    all_idx = []
    for group in scaffold_groups:
        all_idx.extend(group)

    n = len(all_idx)
    n_train = int(n * frac_train)
    n_val = int(n * frac_val)

    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:n_train + n_val]
    test_idx = all_idx[n_train + n_val:]

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def _download_csv(url: str, cache_dir: str = "/tmp/marl_mol_data") -> str:
    """Download a CSV file to cache, return local path."""
    import os
    import urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    filename = url.split("/")[-1].split("?")[0]
    local_path = os.path.join(cache_dir, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename}...", end=" ", flush=True)
        urllib.request.urlretrieve(url, local_path)
        print("done")
    return local_path


def load_esol(seed: int = 42) -> MolecularDataset:
    """Load ESOL (aqueous solubility) dataset."""
    import pandas as pd
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
    path = _download_csv(url)
    df = pd.read_csv(path)
    smiles_list = df["smiles"].tolist()
    targets_raw = df["measured log solubility in mols per litre"].values

    return _build_molecular_dataset(
        name="ESOL",
        task_type="regression",
        smiles_list=smiles_list,
        targets_raw=targets_raw,
        seed=seed,
    )


def load_freesolv(seed: int = 42) -> MolecularDataset:
    """Load FreeSolv (hydration free energy) dataset."""
    import pandas as pd
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv"
    path = _download_csv(url)
    df = pd.read_csv(path)
    smiles_list = df["smiles"].tolist()
    targets_raw = df["expt"].values

    return _build_molecular_dataset(
        name="FreeSolv",
        task_type="regression",
        smiles_list=smiles_list,
        targets_raw=targets_raw,
        seed=seed,
    )


def load_lipophilicity(seed: int = 42) -> MolecularDataset:
    """Load Lipophilicity dataset."""
    import pandas as pd
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
    path = _download_csv(url)
    df = pd.read_csv(path)
    smiles_list = df["smiles"].tolist()
    targets_raw = df["exp"].values

    return _build_molecular_dataset(
        name="Lipophilicity",
        task_type="regression",
        smiles_list=smiles_list,
        targets_raw=targets_raw,
        seed=seed,
    )


def load_bbbp(seed: int = 42) -> MolecularDataset:
    """Load BBBP (blood-brain barrier penetration) classification dataset."""
    import pandas as pd
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
    path = _download_csv(url)
    df = pd.read_csv(path)
    smiles_list = df["smiles"].tolist()
    targets_raw = df["p_np"].values.astype(float)

    return _build_molecular_dataset(
        name="BBBP",
        task_type="classification",
        smiles_list=smiles_list,
        targets_raw=targets_raw,
        seed=seed,
    )


def _build_molecular_dataset(
    name: str,
    task_type: str,
    smiles_list: list[str],
    targets_raw: np.ndarray,
    seed: int = 42,
    fp_bits: int = 1024,
    max_atoms: int = 60,
) -> MolecularDataset:
    """Build a MolecularDataset from SMILES and targets."""
    from rdkit import Chem

    valid_indices = []
    all_fps = []
    all_descs = []
    all_smiles_enc = []
    all_atom_feats = []
    valid_smiles = []
    valid_targets = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        try:
            fp = compute_morgan_fingerprint(mol, n_bits=fp_bits)
            desc = compute_rdkit_descriptors(mol)
            smi_enc = smiles_to_encoding(smi)
            atom_feat = compute_atom_features(mol, max_atoms=max_atoms)
        except Exception:
            continue

        valid_indices.append(i)
        all_fps.append(fp)
        all_descs.append(desc)
        all_smiles_enc.append(smi_enc)
        all_atom_feats.append(atom_feat)
        valid_smiles.append(smi)
        valid_targets.append(targets_raw[i])

    # Build tensors
    fingerprints = torch.tensor(np.stack(all_fps), dtype=torch.float32)
    descriptors = torch.tensor(np.stack(all_descs), dtype=torch.float32)
    smiles_encoded = torch.tensor(np.stack(all_smiles_enc), dtype=torch.float32)
    atom_features = torch.tensor(np.stack(all_atom_feats), dtype=torch.float32)
    targets = torch.tensor(np.array(valid_targets), dtype=torch.float32).unsqueeze(-1)

    # Normalize descriptors (z-score)
    desc_mean = descriptors.mean(dim=0, keepdim=True)
    desc_std = descriptors.std(dim=0, keepdim=True).clamp(min=1e-8)
    descriptors = (descriptors - desc_mean) / desc_std

    # Normalize targets for regression
    if task_type == "regression":
        t_mean = targets.mean()
        t_std = targets.std().clamp(min=1e-8)
        targets = (targets - t_mean) / t_std

    # Scaffold split
    train_idx, val_idx, test_idx = scaffold_split(valid_smiles, seed=seed)

    return MolecularDataset(
        name=name,
        task_type=task_type,
        fingerprints=fingerprints,
        descriptors=descriptors,
        smiles_encoded=smiles_encoded,
        atom_features=atom_features,
        targets=targets,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        smiles=valid_smiles,
    )


# Dataset loaders registry
DATASETS = {
    "esol": load_esol,
    "freesolv": load_freesolv,
    "lipophilicity": load_lipophilicity,
    "bbbp": load_bbbp,
}


def get_representation_dims(dataset: MolecularDataset) -> dict:
    """Get input dimensions for each representation."""
    return {
        "fingerprint": dataset.fingerprints.shape[1],
        "descriptor": dataset.descriptors.shape[1],
        "smiles": dataset.smiles_encoded.shape[1],
        "atom": dataset.atom_features.shape[1],
    }
