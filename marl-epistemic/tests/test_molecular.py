"""Tests for molecular environment and calibrated evidence."""

import pytest
import torch
import numpy as np

from src.agents import LinearAgent, MLPAgent


class TestCalibratedEvidence:
    """Test the calibrated evidence method (mc_dropout * validation accuracy)."""

    def test_calibrated_evidence_positive(self):
        agent = MLPAgent(20, 1)
        x = torch.randn(8, 20)
        # Calibrate with some dummy validation data
        x_val = torch.randn(16, 20)
        y_val = torch.randn(16, 1)
        agent.calibrate_evidence(x_val, y_val)
        v = agent.weight_of_evidence(x, method="calibrated")
        assert v.shape == (8,)
        assert (v > 0).all()

    def test_calibration_scale_reflects_accuracy(self):
        """Better models should have higher calibration scale."""
        agent_good = LinearAgent(10, 1)
        agent_bad = LinearAgent(10, 1)

        x_val = torch.randn(32, 10)
        # "Good" agent: targets close to predictions
        with torch.no_grad():
            y_good = agent_good.predict(x_val) + torch.randn(32, 1) * 0.01
            y_bad = torch.randn(32, 1) * 10  # random targets

        agent_good.calibrate_evidence(x_val, y_good)
        agent_bad.calibrate_evidence(x_val, y_bad)

        assert agent_good._calibration_scale > agent_bad._calibration_scale

    def test_calibrated_vs_raw_reweighting(self):
        """Calibrated evidence should reweight relative to raw mc_dropout."""
        agent1 = MLPAgent(20, 1, dropout_rate=0.2)
        agent2 = MLPAgent(20, 1, dropout_rate=0.2)

        x = torch.randn(8, 20)
        x_val = torch.randn(16, 20)

        # Agent 1: well calibrated (low val error)
        with torch.no_grad():
            y_val1 = agent1.predict(x_val) + torch.randn(16, 1) * 0.01
        agent1.calibrate_evidence(x_val, y_val1)

        # Agent 2: poorly calibrated (high val error)
        y_val2 = torch.randn(16, 1) * 5
        agent2.calibrate_evidence(x_val, y_val2)

        # Raw MC dropout may give similar values
        raw1 = agent1.weight_of_evidence(x, method="mc_dropout")
        raw2 = agent2.weight_of_evidence(x, method="mc_dropout")

        # But calibrated should favor agent1
        cal1 = agent1.weight_of_evidence(x, method="calibrated")
        cal2 = agent2.weight_of_evidence(x, method="calibrated")

        # Ratio should shift toward agent1 after calibration
        raw_ratio = raw1.mean() / raw2.mean()
        cal_ratio = cal1.mean() / cal2.mean()
        assert cal_ratio > raw_ratio

    def test_uncalibrated_defaults_to_scale_1(self):
        """Without calibration, calibration_scale should be 1.0."""
        agent = MLPAgent(20, 1)
        assert agent._calibration_scale == 1.0


class TestMolecularEnvironment:
    """Test molecular dataset loading and representations (requires rdkit)."""

    @pytest.fixture
    def has_rdkit(self):
        try:
            from rdkit import Chem
            return True
        except ImportError:
            pytest.skip("rdkit not installed")

    def test_smiles_encoding(self, has_rdkit):
        from src.environments.molecular import smiles_to_encoding, MAX_SMILES_LEN
        enc = smiles_to_encoding("CCO")
        assert len(enc) == MAX_SMILES_LEN
        assert enc[0] > 0  # first char encoded
        assert enc[3] == 0  # padding

    def test_morgan_fingerprint(self, has_rdkit):
        from rdkit import Chem
        from src.environments.molecular import compute_morgan_fingerprint
        mol = Chem.MolFromSmiles("CCO")
        fp = compute_morgan_fingerprint(mol, n_bits=1024)
        assert fp.shape == (1024,)
        assert fp.sum() > 0

    def test_rdkit_descriptors(self, has_rdkit):
        from rdkit import Chem
        from src.environments.molecular import compute_rdkit_descriptors
        mol = Chem.MolFromSmiles("CCO")
        desc = compute_rdkit_descriptors(mol)
        assert desc.shape == (24,)
        assert np.isfinite(desc).all()

    def test_atom_features(self, has_rdkit):
        from rdkit import Chem
        from src.environments.molecular import compute_atom_features
        mol = Chem.MolFromSmiles("CCO")
        feat = compute_atom_features(mol, max_atoms=60)
        assert feat.shape == (600,)  # 60 * 10

    def test_scaffold_split(self, has_rdkit):
        from src.environments.molecular import scaffold_split
        smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "c1ccc(O)cc1",
                   "CC=O", "CCCO", "c1ccc(N)cc1"]
        train, val, test = scaffold_split(smiles, frac_train=0.6, frac_val=0.2)
        # All indices covered
        all_idx = set(train.tolist() + val.tolist() + test.tolist())
        assert all_idx == set(range(len(smiles)))
        # No overlap
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0

    def test_get_representation_dims(self, has_rdkit):
        from src.environments.molecular import get_representation_dims, MolecularDataset
        ds = MolecularDataset(
            name="test",
            task_type="regression",
            fingerprints=torch.zeros(10, 1024),
            descriptors=torch.zeros(10, 24),
            smiles_encoded=torch.zeros(10, 120),
            atom_features=torch.zeros(10, 600),
            targets=torch.zeros(10, 1),
            train_idx=np.arange(7),
            val_idx=np.arange(7, 8),
            test_idx=np.arange(8, 10),
            smiles=["C"] * 10,
        )
        dims = get_representation_dims(ds)
        assert dims == {
            "fingerprint": 1024,
            "descriptor": 24,
            "smiles": 120,
            "atom": 600,
        }
