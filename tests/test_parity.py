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
