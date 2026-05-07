"""Tests for proteon-pyg.

Tests skip when torch_geometric is not installed (matches the optional
[pyg] extra). The proteon import itself is always required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from proteon_pyg import (
    DSSP_CLASSES,
    decode_dssp,
    encode_dssp,
)


TEST_PDB = Path("/scratch/TMAlign/proteon/test-pdbs/1crn.pdb")
TEST_PDB_HET = Path("/scratch/TMAlign/proteon/test-pdbs/1ake.pdb")


# ---------------------------------------------------------------------------
# DSSP int encoding round-trip — does not need torch_geometric, only torch.
# ---------------------------------------------------------------------------


def test_dssp_classes_are_8_state() -> None:
    assert len(DSSP_CLASSES) == 8
    assert set(DSSP_CLASSES) == set("HBEGITSC")


def test_encode_dssp_round_trip() -> None:
    pytest.importorskip("torch")
    s = "HHHHEEEECCCSSGGGITT"
    codes = encode_dssp(s)
    assert decode_dssp(codes) == s


def test_encode_dssp_unknown_chars_decode_question_mark() -> None:
    pytest.importorskip("torch")
    codes = encode_dssp("HX?C")
    assert decode_dssp(codes) == "H??C"


# ---------------------------------------------------------------------------
# proteon_pyg_data — happy path on 1crn (pure protein)
# ---------------------------------------------------------------------------


def _import_pyg():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from proteon_pyg import proteon_pyg_data

    return proteon_pyg_data


@pytest.mark.skipif(not TEST_PDB.exists(), reason="1crn.pdb fixture not available")
def test_proteon_pyg_data_residue_level_1crn() -> None:
    proteon_pyg_data = _import_pyg()
    data = proteon_pyg_data(TEST_PDB, hbond_count=True, dihedrals=True)

    n = data.pos.shape[0]
    assert data.pos.shape == (n, 3)
    assert len(data.chain_id) == n
    assert data.residue_number.shape == (n,)
    assert data.is_amino_acid.shape == (n,)
    assert data.is_amino_acid.all(), "1crn is pure protein — every residue should be AA"

    assert data.residue_sasa.shape == (n,)
    assert data.rsa.shape == (n,)
    assert data.dssp.shape == (n,)
    assert (data.dssp >= 0).all(), "no -1 expected — every residue is AA"

    assert data.hbond_count.shape == (n,)
    assert (data.hbond_count >= 0).all(), "all AA, no -1 sentinel expected"

    assert data.phi.shape == (n,)
    assert data.psi.shape == (n,)
    assert data.omega.shape == (n,)

    # phi NaN at N-terminus, psi NaN at C-terminus, omega NaN at N-terminus.
    import torch

    n_nan_phi = torch.isnan(data.phi).sum().item()
    n_nan_psi = torch.isnan(data.psi).sum().item()
    assert n_nan_phi == 1
    assert n_nan_psi == 1


@pytest.mark.skipif(not TEST_PDB.exists(), reason="1crn.pdb fixture not available")
def test_proteon_pyg_data_energy_attached() -> None:
    proteon_pyg_data = _import_pyg()
    data = proteon_pyg_data(TEST_PDB)

    assert hasattr(data, "proteon_energy_total")
    assert data.proteon_energy_total.dim() == 0
    assert hasattr(data, "proteon_ff")
    assert data.proteon_ff == "charmm19_eef1"


@pytest.mark.skipif(not TEST_PDB.exists(), reason="1crn.pdb fixture not available")
def test_proteon_pyg_data_selective_flags() -> None:
    proteon_pyg_data = _import_pyg()
    data = proteon_pyg_data(
        TEST_PDB,
        sasa=False,
        dssp=False,
        energy=False,
        hbond_count=True,
        dihedrals=False,
    )
    assert not hasattr(data, "residue_sasa")
    assert not hasattr(data, "dssp")
    assert not hasattr(data, "phi")
    assert not hasattr(data, "proteon_energy_total")
    assert hasattr(data, "hbond_count")


# ---------------------------------------------------------------------------
# Mixed-residue 1ake — non-AA residues get DSSP=-1, hbond_count=-1.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TEST_PDB_HET.exists(), reason="1ake.pdb fixture not available")
def test_proteon_pyg_data_handles_hetatm_1ake() -> None:
    proteon_pyg_data = _import_pyg()
    data = proteon_pyg_data(TEST_PDB_HET, hbond_count=True, dihedrals=True)

    n_aa = int(data.is_amino_acid.sum().item())
    n_total = data.pos.shape[0]
    assert n_aa < n_total, "1ake has waters/ligand residues alongside AAs"

    # Every non-AA residue must have dssp == -1 and hbond_count == -1.
    non_aa_mask = ~data.is_amino_acid
    assert (data.dssp[non_aa_mask] == -1).all()
    assert (data.hbond_count[non_aa_mask] == -1).all()

    # Every AA residue must have dssp >= 0.
    assert (data.dssp[data.is_amino_acid] >= 0).all()


# ---------------------------------------------------------------------------
# ProteonFeatures transform
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TEST_PDB.exists(), reason="1crn.pdb fixture not available")
def test_proteon_features_transform_attaches_features() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    from proteon_pyg import ProteonFeatures

    data = Data()
    data.pdb_path = str(TEST_PDB)
    tf = ProteonFeatures(hbond_count=True, dihedrals=True)
    out = tf(data)

    assert out is data, "transform must mutate and return the same Data instance"
    assert hasattr(out, "residue_sasa")
    assert hasattr(out, "dssp")
    assert hasattr(out, "phi")
    assert hasattr(out, "proteon_energy_total")


def test_proteon_features_transform_requires_pdb_path() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    from proteon_pyg import ProteonFeatures

    data = Data()
    tf = ProteonFeatures()
    with pytest.raises(ValueError, match="pdb_path"):
        tf(data)
