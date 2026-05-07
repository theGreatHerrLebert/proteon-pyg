"""EVIDENT parity claim: PyG Data tensor values equal direct proteon outputs.

Manifest: ../evident.yaml#proteon-pyg-residue-feature-parity

The science claim of this package is "proteon residue features attached as
PyTorch Geometric Data tensors are the same values proteon would return when
called directly." If that doesn't hold, every downstream geometric-DL model
trained on a proteon-pyg pipeline inherits a silent bias.

Oracle: proteon itself, called on the same PDB.
Tolerance: exact equality (post .numpy() round-trip) for SASA / RSA / energy
floats; integer equality for DSSP int codes and hbond_count; NaN-aware for
RSA on non-standard residues and phi/psi/omega at chain termini (NaN must
remain NaN at the same indices).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import proteon


TEST_PDB = Path("/scratch/TMAlign/proteon/test-pdbs/1crn.pdb")
TEST_PDB_HET = Path("/scratch/TMAlign/proteon/test-pdbs/1ake.pdb")


def _import_pyg():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from proteon_pyg import DSSP_CLASSES, proteon_pyg_data

    return proteon_pyg_data, DSSP_CLASSES


@pytest.mark.skipif(not TEST_PDB.exists(), reason="1crn.pdb fixture not available")
def test_parity_residue_features_match_direct_proteon_call() -> None:
    """Every tensor index must equal the corresponding direct-API value."""
    proteon_pyg_data, DSSP_CLASSES = _import_pyg()

    data = proteon_pyg_data(TEST_PDB, hbond_count=True, dihedrals=True)
    structure = proteon.load(str(TEST_PDB))

    rs_oracle = np.asarray(proteon.residue_sasa(structure), dtype=np.float32)
    rsa_oracle = np.asarray(proteon.relative_sasa(structure), dtype=np.float32)
    dssp_oracle = proteon.dssp(structure)
    hbond_oracle = np.asarray(proteon.hbond_count(structure), dtype=np.int64)
    phi_o, psi_o, omega_o = proteon.backbone_dihedrals(structure)

    # Exact float-equality for SASA, RSA via numpy round-trip.
    np.testing.assert_array_equal(data.residue_sasa.numpy(), rs_oracle)
    np.testing.assert_array_equal(data.rsa.numpy(), rsa_oracle)

    # DSSP: tensor int codes must decode back to the proteon string at
    # the AA-only positions.
    aa_indices = np.flatnonzero(data.is_amino_acid.numpy())
    code_to_char = {i: c for i, c in enumerate(DSSP_CLASSES)}
    decoded = "".join(code_to_char[int(data.dssp[i].item())] for i in aa_indices)
    assert decoded == dssp_oracle, (
        f"DSSP tensor decoded mismatch: tensor={decoded!r} oracle={dssp_oracle!r}"
    )

    # hbond_count at AA positions must equal the proteon array.
    hbond_at_aa = data.hbond_count.numpy()[aa_indices]
    np.testing.assert_array_equal(hbond_at_aa, hbond_oracle)

    # phi/psi/omega at AA positions must equal proteon's per-AA arrays.
    # NaN positions must be NaN in both.
    for name, oracle in (("phi", phi_o), ("psi", psi_o), ("omega", omega_o)):
        tensor_at_aa = getattr(data, name).numpy()[aa_indices]
        oracle_arr = np.asarray(oracle, dtype=np.float32)
        # Compare with NaN equality.
        for i, (t, o) in enumerate(zip(tensor_at_aa, oracle_arr)):
            if math.isnan(o):
                assert math.isnan(t), (
                    f"{name}[{i}] NaN-parity broken: tensor={t!r} oracle={o!r}"
                )
            else:
                assert t == o, (
                    f"{name}[{i}] mismatch: tensor={t!r} oracle={o!r}"
                )


@pytest.mark.skipif(not TEST_PDB.exists(), reason="1crn.pdb fixture not available")
def test_parity_energy_dict_matches_direct_proteon_call() -> None:
    """Each proteon_energy_<component> tensor must equal the direct dict entry."""
    proteon_pyg_data, _ = _import_pyg()

    data = proteon_pyg_data(TEST_PDB)
    structure = proteon.load(str(TEST_PDB))
    oracle = proteon.compute_energy(structure, ff="charmm19_eef1")

    n_checked = 0
    for k, v in oracle.items():
        if isinstance(v, (int, float)):
            attr = f"proteon_energy_{k}"
            assert hasattr(data, attr), f"missing {attr} on Data"
            tensor_val = float(getattr(data, attr).item())
            assert tensor_val == float(v), (
                f"{attr} mismatch: tensor={tensor_val!r} oracle={v!r}"
            )
            n_checked += 1
    assert n_checked > 0, "no energy components were checked"


@pytest.mark.skipif(not TEST_PDB_HET.exists(), reason="1ake.pdb fixture not available")
def test_parity_hetatm_residues_dssp_minus_one_for_non_aa() -> None:
    """On a PDB with HETATMs, every non-AA residue must have dssp == -1."""
    proteon_pyg_data, _ = _import_pyg()
    data = proteon_pyg_data(TEST_PDB_HET, hbond_count=True)

    non_aa = ~data.is_amino_acid
    if non_aa.sum().item() == 0:
        pytest.skip("1ake fixture had no non-AA residues — unexpected, but skip cleanly")
    assert (data.dssp[non_aa] == -1).all()
    assert (data.hbond_count[non_aa] == -1).all()


# ---------------------------------------------------------------------------
# Atom-level parity (v0.0.2)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TEST_PDB.exists(), reason="1crn.pdb fixture not available")
def test_parity_atom_level_features_match_direct_proteon_call() -> None:
    """Atom-level tensor values must equal direct proteon API outputs."""
    proteon_pyg_data, _ = _import_pyg()

    data = proteon_pyg_data(TEST_PDB, granularity="atom")
    structure = proteon.load(str(TEST_PDB))

    # atom_sasa exact equality
    atom_sasa_oracle = np.asarray(proteon.atom_sasa(structure), dtype=np.float32)
    np.testing.assert_array_equal(data.atom_sasa.numpy(), atom_sasa_oracle)

    # Per-atom charge / is_backbone / hetero from the Atom API, in the
    # residues -> residue.atoms flat order the adapter uses.
    charges = []
    is_bb = []
    hetero = []
    coords = []
    atom_names = []
    elements = []
    residue_idx_oracle = []
    for r_i, residue in enumerate(structure.residues):
        for atom in residue.atoms:
            charges.append(float(atom.charge))
            is_bb.append(bool(atom.is_backbone))
            hetero.append(bool(atom.hetero))
            coords.append((atom.x, atom.y, atom.z))
            atom_names.append(atom.name)
            elements.append(atom.element or "")
            residue_idx_oracle.append(r_i)

    np.testing.assert_array_equal(
        data.charge.numpy(), np.asarray(charges, dtype=np.float32)
    )
    assert data.is_backbone.tolist() == is_bb
    assert data.hetero.tolist() == hetero
    np.testing.assert_array_equal(
        data.pos.numpy(), np.asarray(coords, dtype=np.float32)
    )
    assert data.atom_name == atom_names
    assert data.element == elements
    np.testing.assert_array_equal(
        data.residue_index.numpy(),
        np.asarray(residue_idx_oracle, dtype=np.int64),
    )

    # Residue features at residue resolution still match the oracle.
    rs_oracle = np.asarray(proteon.residue_sasa(structure), dtype=np.float32)
    np.testing.assert_array_equal(data.residue_sasa.numpy(), rs_oracle)


# ---------------------------------------------------------------------------
# Batch parity (v0.0.3)
# ---------------------------------------------------------------------------


def _data_attrs_equal(a, b) -> None:
    """Assert two Data objects have identical attribute sets and values."""
    import torch

    a_keys = set(a.to_dict().keys())
    b_keys = set(b.to_dict().keys())
    assert a_keys == b_keys, f"attr key sets differ: {a_keys ^ b_keys}"
    for k in a_keys:
        va = getattr(a, k)
        vb = getattr(b, k)
        if isinstance(va, torch.Tensor):
            assert isinstance(vb, torch.Tensor), f"{k}: type mismatch"
            if va.dtype.is_floating_point:
                np.testing.assert_array_equal(va.numpy(), vb.numpy())
            else:
                assert torch.equal(va, vb), f"{k}: tensor values differ"
        else:
            assert va == vb, f"{k}: value mismatch ({va!r} != {vb!r})"


@pytest.mark.skipif(
    not (TEST_PDB.exists() and TEST_PDB_HET.exists()),
    reason="1crn.pdb and/or 1ake.pdb fixture not available",
)
def test_parity_batch_helper_matches_single_call_loop_residue() -> None:
    """proteon_pyg_data_batch must produce identical Data to a single-call loop."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from proteon_pyg import proteon_pyg_data, proteon_pyg_data_batch

    paths = [TEST_PDB, TEST_PDB_HET, TEST_PDB]
    batch = proteon_pyg_data_batch(paths, hbond_count=True, dihedrals=True)
    serial = [
        proteon_pyg_data(p, hbond_count=True, dihedrals=True) for p in paths
    ]
    assert len(batch) == len(serial)
    for b, s in zip(batch, serial):
        _data_attrs_equal(b, s)


@pytest.mark.skipif(not TEST_PDB.exists(), reason="1crn.pdb fixture not available")
def test_parity_batch_helper_matches_single_call_loop_atom() -> None:
    """Batch parity holds at atom granularity too."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from proteon_pyg import proteon_pyg_data, proteon_pyg_data_batch

    paths = [TEST_PDB, TEST_PDB]
    batch = proteon_pyg_data_batch(
        paths, granularity="atom", hbond_count=True, dihedrals=True
    )
    serial = [
        proteon_pyg_data(p, granularity="atom", hbond_count=True, dihedrals=True)
        for p in paths
    ]
    for b, s in zip(batch, serial):
        _data_attrs_equal(b, s)
