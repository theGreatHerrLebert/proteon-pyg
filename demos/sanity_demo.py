"""Sanity demo: do proteon-pyg's tensors look like real protein features?

Loads a handful of curated PDBs through ``proteon_pyg_data_batch``, checks
each tensor lands in the physical range you'd expect from biology, and
emits plots so a reviewer can eyeball the distributions.

This is a *sanity* demo, not a science claim. We are NOT showing that
proteon features improve any downstream model. We're showing that the
adapter's tensors are physically plausible and NaN-correct -- which is
the floor any science demo must clear before it's worth building.

Run:
    /scratch/TMAlign/proteon/.venv/bin/python demos/sanity_demo.py

Outputs land in ``demos/output/sanity/``. Console output ends in PASS
when every range check holds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

from proteon_pyg import DSSP_CLASSES, proteon_pyg_data_batch


FIXTURE_DIR = Path("/scratch/TMAlign/proteon/test-pdbs")
PDB_NAMES = ["1crn.pdb", "1aaj.pdb", "1ake.pdb", "1bpi.pdb", "1ubq.pdb"]
OUTPUT_DIR = Path(__file__).parent / "output" / "sanity"


def _gather_pdbs() -> list[Path]:
    paths = [FIXTURE_DIR / n for n in PDB_NAMES]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"[sanity] missing fixtures: {missing}", file=sys.stderr)
        sys.exit(2)
    return paths


def _check_ranges(datas: list) -> list[str]:
    """Return a list of sanity-violation messages (empty if all pass)."""
    failures: list[str] = []
    for d, name in zip(datas, PDB_NAMES):
        # SASA non-negative.
        if (d.residue_sasa < 0).any().item():
            failures.append(f"{name}: residue_sasa has negative values")
        # RSA NaN allowed for non-AA; AA positions should mostly be in [0, 1.5].
        rsa_aa = d.rsa[d.is_amino_acid]
        rsa_finite = rsa_aa[~torch.isnan(rsa_aa)]
        if rsa_finite.numel() and (rsa_finite < 0).any().item():
            failures.append(f"{name}: rsa < 0 at AA positions")
        # phi/psi/omega in [-180, 180] where defined.
        for attr in ("phi", "psi", "omega"):
            if not hasattr(d, attr):
                continue
            v = getattr(d, attr)
            v_finite = v[~torch.isnan(v)]
            if v_finite.numel() == 0:
                continue
            if (v_finite < -180.001).any().item() or (v_finite > 180.001).any().item():
                failures.append(f"{name}: {attr} out of [-180, 180]")
        # DSSP int code in [-1, 7].
        if (d.dssp < -1).any().item() or (d.dssp > 7).any().item():
            failures.append(f"{name}: dssp out of [-1, 7]")
        # NaN positions: phi NaN at exactly N-terminus per chain, psi NaN at
        # C-terminus. On 1crn (single chain) that's 1 NaN each.
        if name == "1crn.pdb":
            n_phi_nan = int(torch.isnan(d.phi).sum().item())
            n_psi_nan = int(torch.isnan(d.psi).sum().item())
            if n_phi_nan != 1 or n_psi_nan != 1:
                failures.append(
                    f"{name}: expected 1 NaN at chain termini for phi/psi, "
                    f"got phi={n_phi_nan} psi={n_psi_nan}"
                )
    return failures


def _plot_ramachandran(datas, out: Path) -> None:
    phi_all = []
    psi_all = []
    for d in datas:
        if hasattr(d, "phi"):
            mask = ~(torch.isnan(d.phi) | torch.isnan(d.psi))
            phi_all.extend(d.phi[mask].tolist())
            psi_all.extend(d.psi[mask].tolist())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(phi_all, psi_all, s=4, alpha=0.6)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel(r"$\varphi$ (deg)")
    ax.set_ylabel(r"$\psi$ (deg)")
    ax.set_title("Ramachandran (proteon-pyg)")
    ax.axhline(0, color="0.7", linewidth=0.5)
    ax.axvline(0, color="0.7", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_sasa_rsa(datas, out: Path) -> None:
    sasa_all = np.concatenate([d.residue_sasa.numpy() for d in datas])
    rsa_all = np.concatenate([d.rsa.numpy() for d in datas])
    rsa_finite = rsa_all[~np.isnan(rsa_all)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(sasa_all, bins=40)
    axes[0].set_xlabel(r"residue SASA ($\AA^2$)")
    axes[0].set_ylabel("count")
    axes[0].set_title("residue_sasa across 5 PDBs")
    axes[1].hist(rsa_finite, bins=40)
    axes[1].set_xlabel("relative SASA")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"rsa (NaN-filtered, n={len(rsa_finite)})")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_dssp_histogram(datas, out: Path) -> None:
    counts = np.zeros(len(DSSP_CLASSES), dtype=np.int64)
    n_non_aa = 0
    for d in datas:
        codes = d.dssp.numpy()
        for c in codes:
            if 0 <= c < len(DSSP_CLASSES):
                counts[c] += 1
            else:
                n_non_aa += 1

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(DSSP_CLASSES, counts)
    ax.set_xlabel("DSSP-8 class")
    ax.set_ylabel("residues")
    ax.set_title(f"DSSP class distribution (5 PDBs, {n_non_aa} non-AA skipped)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_energy_components(datas, out: Path) -> None:
    components = [
        "bond_stretch",
        "angle_bend",
        "torsion",
        "improper_torsion",
        "vdw",
        "electrostatic",
        "solvation",
        "total",
    ]
    n = len(datas)
    bar_x = np.arange(len(components))
    width = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(11, 4))
    for i, (d, name) in enumerate(zip(datas, PDB_NAMES)):
        vals = [
            float(getattr(d, f"proteon_energy_{c}").item()) for c in components
        ]
        ax.bar(bar_x + i * width, vals, width=width, label=name)
    ax.set_xticks(bar_x + width * (n - 1) / 2)
    ax.set_xticklabels(components, rotation=30, ha="right")
    ax.set_ylabel("energy (kJ/mol)")
    ax.set_title("CHARMM19+EEF1 energy components per PDB")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = _gather_pdbs()
    print(f"[sanity] loading {len(paths)} PDBs through proteon_pyg_data_batch...")
    datas = proteon_pyg_data_batch(paths, hbond_count=True, dihedrals=True)

    print("[sanity] tensor shapes:")
    for d, name in zip(datas, PDB_NAMES):
        print(
            f"  {name}: pos={tuple(d.pos.shape)} dssp={tuple(d.dssp.shape)} "
            f"E_total={float(d.proteon_energy_total.item()):.2f} kJ/mol"
        )

    print("[sanity] generating plots...")
    _plot_ramachandran(datas, OUTPUT_DIR / "ramachandran.png")
    _plot_sasa_rsa(datas, OUTPUT_DIR / "sasa_rsa.png")
    _plot_dssp_histogram(datas, OUTPUT_DIR / "dssp.png")
    _plot_energy_components(datas, OUTPUT_DIR / "energy.png")

    print("[sanity] running range checks...")
    failures = _check_ranges(datas)
    if failures:
        print("[sanity] FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"[sanity] PASS — {len(paths)} PDBs, plots in {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
