"""Build PyTorch Geometric Data tensors from proteon structural features.

The trust property mirrors proteon-graphein: every value attached to the Data
object is exact-equal (post NumPy round-trip) to what proteon would return
when called directly. proteon is the oracle.

Single-PDB, residue-level only in v0.0.1. Atom-level (v0.0.2) and a batch
helper (v0.0.3) follow.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import proteon

if TYPE_CHECKING:
    import torch
    from torch_geometric.data import Data


# 8-state DSSP classes, encoded as integers 0..7. Non-AA residues get -1.
# Order is the canonical DSSP-8 alphabet used by proteon's dssp() output.
DSSP_CLASSES: tuple[str, ...] = ("H", "B", "E", "G", "I", "T", "S", "C")
_DSSP_INDEX: dict[str, int] = {c: i for i, c in enumerate(DSSP_CLASSES)}


def encode_dssp(dssp_str: str) -> "torch.Tensor":
    """Encode a DSSP-8 string into a (len,) long tensor.

    Each character in ``DSSP_CLASSES`` maps to its index (0..7). Any
    character not in the alphabet maps to -1. The reverse is :func:`decode_dssp`.
    """
    import torch

    codes = np.fromiter(
        (_DSSP_INDEX.get(c, -1) for c in dssp_str),
        dtype=np.int64,
        count=len(dssp_str),
    )
    return torch.from_numpy(codes)


def decode_dssp(codes: "torch.Tensor") -> str:
    """Decode an int tensor of DSSP indices back to its string form.

    -1 (and any out-of-range value) decodes to '?'. The round-trip
    ``decode_dssp(encode_dssp(s)) == s`` holds when ``s`` only contains
    canonical DSSP-8 characters.
    """
    arr = codes.detach().cpu().numpy()
    out = []
    for v in arr.tolist():
        if 0 <= v < len(DSSP_CLASSES):
            out.append(DSSP_CLASSES[v])
        else:
            out.append("?")
    return "".join(out)


def _ca_position(residue) -> tuple[float, float, float]:
    """Return the residue's CA coords, or (NaN, NaN, NaN) if absent.

    Falls back to the first atom of the residue for residues without a CA
    (waters, ligands, nucleotides) so that ``data.pos`` is always populated.
    """
    for atom in residue.atoms:
        if atom.name.strip() == "CA":
            return atom.x, atom.y, atom.z
    for atom in residue.atoms:
        return atom.x, atom.y, atom.z
    return float("nan"), float("nan"), float("nan")


def _build_residue_tensors(
    structure,
    feats: dict[str, Any],
    *,
    sasa: bool,
    dssp: bool,
    hbond_count: bool,
    dihedrals: bool,
) -> dict[str, "torch.Tensor | list"]:
    """Build per-residue tensors aligned to ``structure.residues`` order.

    Length-N (N = total residue count, including non-AA). DSSP indices and
    AA-only features are padded with -1 / NaN at non-AA positions.
    """
    import torch

    residues = list(structure.residues)
    n = len(residues)

    pos = np.empty((n, 3), dtype=np.float32)
    chain_id: list[str] = []
    residue_number = np.empty(n, dtype=np.int64)
    insertion_code: list[str] = []
    residue_name: list[str] = []
    is_amino_acid = np.empty(n, dtype=bool)

    aa_idx = -np.ones(n, dtype=np.int64)
    aa_counter = 0
    for i, r in enumerate(residues):
        x, y, z = _ca_position(r)
        pos[i] = (x, y, z)
        chain_id.append(r.chain_id or "")
        residue_number[i] = r.serial_number
        insertion_code.append("" if r.insertion_code is None else str(r.insertion_code).strip())
        residue_name.append((r.name or "").strip())
        is_aa = bool(r.is_amino_acid)
        is_amino_acid[i] = is_aa
        if is_aa:
            aa_idx[i] = aa_counter
            aa_counter += 1

    out: dict[str, Any] = {
        "pos": torch.from_numpy(pos),
        "chain_id": chain_id,
        "residue_number": torch.from_numpy(residue_number),
        "insertion_code": insertion_code,
        "residue_name": residue_name,
        "is_amino_acid": torch.from_numpy(is_amino_acid),
    }

    if sasa:
        rs = np.asarray(feats["residue_sasa"], dtype=np.float32)
        rsa = np.asarray(feats["rsa"], dtype=np.float32)
        if len(rs) != n or len(rsa) != n:
            raise RuntimeError(
                f"proteon SASA arrays length {len(rs)} / RSA {len(rsa)} != residue count {n}"
            )
        out["residue_sasa"] = torch.from_numpy(rs)
        out["rsa"] = torch.from_numpy(rsa)

    if dssp:
        dssp_str = feats["dssp"]
        if len(dssp_str) != aa_counter:
            raise RuntimeError(
                f"proteon DSSP string length {len(dssp_str)} != AA count {aa_counter}"
            )
        codes = -np.ones(n, dtype=np.int64)
        for i, j in enumerate(aa_idx):
            if j >= 0:
                c = dssp_str[j]
                codes[i] = _DSSP_INDEX.get(c, -1)
        out["dssp"] = torch.from_numpy(codes)

    if hbond_count:
        counts_aa = np.asarray(feats["hbond_count"], dtype=np.int64)
        if len(counts_aa) != aa_counter:
            raise RuntimeError(
                f"proteon hbond_count length {len(counts_aa)} != AA count {aa_counter}"
            )
        # -1 sentinel for non-AA so callers can mask consistently.
        counts = -np.ones(n, dtype=np.int64)
        for i, j in enumerate(aa_idx):
            if j >= 0:
                counts[i] = counts_aa[j]
        out["hbond_count"] = torch.from_numpy(counts)

    if dihedrals:
        for name in ("phi", "psi", "omega"):
            arr = np.asarray(feats[name], dtype=np.float32)
            if len(arr) != aa_counter:
                raise RuntimeError(
                    f"proteon {name} length {len(arr)} != AA count {aa_counter}"
                )
            full = np.full(n, np.nan, dtype=np.float32)
            for i, j in enumerate(aa_idx):
                if j >= 0:
                    full[i] = arr[j]
            out[name] = torch.from_numpy(full)

    return out


def _attach_energy(data: "Data", energy: dict[str, Any], ff: str) -> None:
    """Attach proteon energy components and ff name to a Data object.

    Each numeric component lands as a 0-dim tensor with the prefix
    ``proteon_energy_``; the ff name lands as a string at ``data.proteon_ff``.
    """
    import torch

    for k, v in energy.items():
        if isinstance(v, (int, float)):
            setattr(data, f"proteon_energy_{k}", torch.tensor(float(v), dtype=torch.float64))
    data.proteon_ff = ff


def proteon_pyg_data(
    pdb_path: str | Path,
    *,
    sasa: bool = True,
    dssp: bool = True,
    energy: bool = True,
    hbond_count: bool = False,
    dihedrals: bool = False,
    ff: str = "charmm19_eef1",
    sasa_radii: str = "bondi",
) -> "Data":
    """Build a PyG ``Data`` from a PDB, with proteon features as tensors.

    Residue-level (one node per proteon residue, AA + non-AA). v0.0.1 does
    not yet support atom-level granularity.

    Per-node tensors when their flag is enabled:
        residue_sasa : (N,) float32, Å²
        rsa          : (N,) float32, NaN for non-standard residues
        dssp         : (N,) int64, 0..7 per ``DSSP_CLASSES``, -1 for non-AA
        hbond_count  : (N,) int64, -1 for non-AA
        phi,psi,omega: (N,) float32, NaN at termini and non-AA

    Always populated:
        pos              : (N, 3) float32, CA coords (or first-atom fallback,
                           NaN if the residue has no atoms)
        chain_id         : list[str], length N
        residue_number   : (N,) int64
        insertion_code   : list[str], length N
        residue_name     : list[str], length N
        is_amino_acid    : (N,) bool

    Graph-level when ``energy=True``:
        proteon_energy_<component> : 0-dim float64 tensor for each component
                                     in proteon's energy dict
        proteon_ff                 : the force-field name string
    """
    from torch_geometric.data import Data

    structure = proteon.load(str(pdb_path))

    feat_dict: dict[str, Any] = {}
    if sasa:
        feat_dict["residue_sasa"] = proteon.residue_sasa(structure, radii=sasa_radii)
        feat_dict["rsa"] = proteon.relative_sasa(structure, radii=sasa_radii)
    if dssp:
        feat_dict["dssp"] = proteon.dssp(structure)
    if hbond_count:
        feat_dict["hbond_count"] = proteon.hbond_count(structure)
    if dihedrals:
        phi, psi, omega = proteon.backbone_dihedrals(structure)
        feat_dict["phi"] = phi
        feat_dict["psi"] = psi
        feat_dict["omega"] = omega

    tensors = _build_residue_tensors(
        structure,
        feat_dict,
        sasa=sasa,
        dssp=dssp,
        hbond_count=hbond_count,
        dihedrals=dihedrals,
    )

    data = Data(**tensors)

    if energy:
        e = proteon.compute_energy(structure, ff=ff)
        _attach_energy(data, e, ff)

    return data


class ProteonFeatures:
    """PyG ``BaseTransform`` that attaches proteon features to a ``Data``.

    Each input ``Data`` must carry a ``pdb_path`` attribute (str or Path)
    pointing to the PDB the Data was built from. The transform is
    side-effect-only on the Data: it adds tensor attributes for each enabled
    feature and returns the same object. v0.0.1 supports residue-level Data
    only.

    Use as you would any PyG transform:

        >>> from torch_geometric.transforms import Compose
        >>> from proteon_pyg import ProteonFeatures
        >>> tf = ProteonFeatures(hbond_count=True, dihedrals=True)
        >>> data = tf(data)  # data.pdb_path must be set

    Or in a Dataset pipeline by passing ``transform=tf`` to your dataset.
    """

    def __init__(
        self,
        *,
        sasa: bool = True,
        dssp: bool = True,
        energy: bool = True,
        hbond_count: bool = False,
        dihedrals: bool = False,
        ff: str = "charmm19_eef1",
        sasa_radii: str = "bondi",
    ) -> None:
        self.sasa = sasa
        self.dssp = dssp
        self.energy = energy
        self.hbond_count = hbond_count
        self.dihedrals = dihedrals
        self.ff = ff
        self.sasa_radii = sasa_radii

    def __call__(self, data: "Data") -> "Data":
        path = getattr(data, "pdb_path", None)
        if path is None:
            raise ValueError(
                "ProteonFeatures requires data.pdb_path to be set (str or Path). "
                "Build your Data with the path attached, or use proteon_pyg_data() instead."
            )
        new = proteon_pyg_data(
            path,
            sasa=self.sasa,
            dssp=self.dssp,
            energy=self.energy,
            hbond_count=self.hbond_count,
            dihedrals=self.dihedrals,
            ff=self.ff,
            sasa_radii=self.sasa_radii,
        )
        # Copy proteon-attached attrs onto the user's Data, preserving any
        # attributes they already set.
        for key, value in new.to_dict().items():
            setattr(data, key, value)
        return data

    def __repr__(self) -> str:
        flags = ", ".join(
            f"{k}={getattr(self, k)}"
            for k in ("sasa", "dssp", "energy", "hbond_count", "dihedrals")
        )
        return f"ProteonFeatures({flags}, ff={self.ff!r})"
