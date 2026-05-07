"""Build PyTorch Geometric Data tensors from proteon structural features.

The trust property mirrors proteon-graphein: every value attached to the Data
object is exact-equal (post NumPy round-trip) to what proteon would return
when called directly. proteon is the oracle.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import proteon

if TYPE_CHECKING:
    import torch
    from torch_geometric.data import Data


Granularity = Literal["residue", "atom"]


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
    """Return the residue's CA coords, or first-atom fallback, or NaN."""
    for atom in residue.atoms:
        if atom.name.strip() == "CA":
            return atom.x, atom.y, atom.z
    for atom in residue.atoms:
        return atom.x, atom.y, atom.z
    return float("nan"), float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Per-residue tensor block (used by both granularities)
# ---------------------------------------------------------------------------


def _build_residue_block(
    structure,
    feats: dict[str, Any],
    *,
    sasa: bool,
    dssp: bool,
    hbond_count: bool,
    dihedrals: bool,
    include_pos: bool,
) -> dict[str, Any]:
    """Build per-residue tensors aligned to ``structure.residues`` order.

    Returns a dict of length-N tensors / lists (N = total residue count,
    AA + non-AA). DSSP / hbond_count / dihedrals are padded with -1 / NaN
    at non-AA positions.

    When ``include_pos`` is False, the ``pos`` key is omitted — used in the
    atom-granularity path where ``pos`` lives on atoms instead.
    """
    import torch

    residues = list(structure.residues)
    n = len(residues)

    chain_id: list[str] = []
    residue_number = np.empty(n, dtype=np.int64)
    insertion_code: list[str] = []
    residue_name: list[str] = []
    is_amino_acid = np.empty(n, dtype=bool)
    pos = np.empty((n, 3), dtype=np.float32) if include_pos else None

    aa_idx = -np.ones(n, dtype=np.int64)
    aa_counter = 0
    for i, r in enumerate(residues):
        chain_id.append(r.chain_id or "")
        residue_number[i] = r.serial_number
        insertion_code.append("" if r.insertion_code is None else str(r.insertion_code).strip())
        residue_name.append((r.name or "").strip())
        is_aa = bool(r.is_amino_acid)
        is_amino_acid[i] = is_aa
        if is_aa:
            aa_idx[i] = aa_counter
            aa_counter += 1
        if include_pos:
            x, y, z = _ca_position(r)
            pos[i] = (x, y, z)

    out: dict[str, Any] = {
        "chain_id": chain_id,
        "residue_number": torch.from_numpy(residue_number),
        "insertion_code": insertion_code,
        "residue_name": residue_name,
        "is_amino_acid": torch.from_numpy(is_amino_acid),
    }
    if include_pos:
        out["pos"] = torch.from_numpy(pos)

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
                codes[i] = _DSSP_INDEX.get(dssp_str[j], -1)
        out["dssp"] = torch.from_numpy(codes)

    if hbond_count:
        counts_aa = np.asarray(feats["hbond_count"], dtype=np.int64)
        if len(counts_aa) != aa_counter:
            raise RuntimeError(
                f"proteon hbond_count length {len(counts_aa)} != AA count {aa_counter}"
            )
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


# ---------------------------------------------------------------------------
# Per-atom tensor block (atom granularity only)
# ---------------------------------------------------------------------------


def _build_atom_block(
    structure,
    feats: dict[str, Any],
    *,
    atom_features: bool,
) -> dict[str, Any]:
    """Build per-atom tensors aligned to flat residues -> residue.atoms order.

    Returns a dict including ``pos`` (atom coords), ``residue_index`` (which
    residue each atom belongs to), and the per-atom features (``atom_sasa``,
    ``charge``, ``is_backbone``, ``hetero``) when ``atom_features`` is True.
    Atom names and elements always populate.
    """
    import torch

    residues = list(structure.residues)
    n_atoms_total = sum(len(list(r.atoms)) for r in residues)

    pos = np.empty((n_atoms_total, 3), dtype=np.float32)
    residue_index = np.empty(n_atoms_total, dtype=np.int64)
    atom_name: list[str] = []
    element: list[str] = []
    charge = np.empty(n_atoms_total, dtype=np.float32) if atom_features else None
    is_backbone = np.empty(n_atoms_total, dtype=bool) if atom_features else None
    hetero = np.empty(n_atoms_total, dtype=bool) if atom_features else None

    idx = 0
    for r_i, r in enumerate(residues):
        for atom in r.atoms:
            pos[idx] = (atom.x, atom.y, atom.z)
            residue_index[idx] = r_i
            atom_name.append(atom.name)
            element.append(atom.element or "")
            if atom_features:
                charge[idx] = float(atom.charge)
                is_backbone[idx] = bool(atom.is_backbone)
                hetero[idx] = bool(atom.hetero)
            idx += 1

    out: dict[str, Any] = {
        "pos": torch.from_numpy(pos),
        "residue_index": torch.from_numpy(residue_index),
        "atom_name": atom_name,
        "element": element,
    }

    if atom_features:
        per_atom_sasa = np.asarray(feats.get("atom_sasa", np.array([])), dtype=np.float32)
        if "atom_sasa" in feats:
            if len(per_atom_sasa) != n_atoms_total:
                raise RuntimeError(
                    f"proteon atom_sasa length {len(per_atom_sasa)} != atom count {n_atoms_total}"
                )
            out["atom_sasa"] = torch.from_numpy(per_atom_sasa)
        out["charge"] = torch.from_numpy(charge)
        out["is_backbone"] = torch.from_numpy(is_backbone)
        out["hetero"] = torch.from_numpy(hetero)

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


# ---------------------------------------------------------------------------
# Internals shared by single-call and batch paths
# ---------------------------------------------------------------------------


def _compute_features_for_structure(
    structure,
    *,
    sasa: bool,
    dssp: bool,
    energy: bool,
    atom_sasa: bool,
    hbond_count: bool,
    dihedrals: bool,
    ff: str,
    sasa_radii: str,
) -> dict[str, Any]:
    """Compute proteon features for a single already-loaded structure."""
    out: dict[str, Any] = {}
    if sasa:
        out["residue_sasa"] = proteon.residue_sasa(structure, radii=sasa_radii)
        out["rsa"] = proteon.relative_sasa(structure, radii=sasa_radii)
    if dssp:
        out["dssp"] = proteon.dssp(structure)
    if energy:
        out["energy"] = proteon.compute_energy(structure, ff=ff)
    if atom_sasa:
        out["atom_sasa"] = proteon.atom_sasa(structure, radii=sasa_radii)
    if hbond_count:
        out["hbond_count"] = proteon.hbond_count(structure)
    if dihedrals:
        phi, psi, omega = proteon.backbone_dihedrals(structure)
        out["phi"] = phi
        out["psi"] = psi
        out["omega"] = omega
    return out


def _data_from_structure(
    structure,
    feats: dict[str, Any],
    *,
    granularity: Granularity,
    sasa: bool,
    dssp: bool,
    energy: bool,
    hbond_count: bool,
    dihedrals: bool,
    atom_features: bool,
    ff: str,
) -> "Data":
    """Build a PyG Data from a structure + precomputed feats dict.

    Used by both single-call and batch entry points.
    """
    from torch_geometric.data import Data

    if granularity == "residue":
        block = _build_residue_block(
            structure,
            feats,
            sasa=sasa,
            dssp=dssp,
            hbond_count=hbond_count,
            dihedrals=dihedrals,
            include_pos=True,
        )
        data = Data(**block)
    elif granularity == "atom":
        residue_block = _build_residue_block(
            structure,
            feats,
            sasa=sasa,
            dssp=dssp,
            hbond_count=hbond_count,
            dihedrals=dihedrals,
            include_pos=False,
        )
        atom_block = _build_atom_block(
            structure,
            feats,
            atom_features=atom_features,
        )
        # Atom-level Data carries both blocks. residue_index links atoms -> residues.
        merged = {**residue_block, **atom_block}
        data = Data(**merged)
    else:
        raise ValueError(f"Unknown granularity {granularity!r}; use 'residue' or 'atom'")

    if energy and "energy" in feats:
        _attach_energy(data, feats["energy"], ff)

    return data


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def proteon_pyg_data(
    pdb_path: str | Path,
    *,
    granularity: Granularity = "residue",
    sasa: bool = True,
    dssp: bool = True,
    energy: bool = True,
    hbond_count: bool = False,
    dihedrals: bool = False,
    atom_features: bool = True,
    ff: str = "charmm19_eef1",
    sasa_radii: str = "bondi",
) -> "Data":
    """Build a PyG ``Data`` from a PDB, with proteon features as tensors.

    Granularity:
        ``"residue"`` (default): one node per proteon residue. Tensors
            ``pos``, ``residue_sasa``, ``dssp`` etc. have shape (N_residues,).
        ``"atom"``: dual-resolution Data. Per-atom tensors live alongside
            per-residue tensors. ``data.pos`` and per-atom features have
            shape (N_atoms, ...); residue features keep their natural
            (N_residues, ...) shape. ``data.residue_index`` (N_atoms,)
            links each atom back to its residue.

    Per-residue tensors when their flag is enabled (always present at
    residue granularity; also attached at atom granularity at residue
    resolution):
        residue_sasa : (N_res,) float32, Å²
        rsa          : (N_res,) float32, NaN for non-standard residues
        dssp         : (N_res,) int64, 0..7 per ``DSSP_CLASSES``, -1 for non-AA
        hbond_count  : (N_res,) int64, -1 for non-AA
        phi,psi,omega: (N_res,) float32, NaN at termini and non-AA

    Atom-only tensors (atom granularity only):
        pos          : (N_atoms, 3) float32, atom coordinates
        atom_sasa    : (N_atoms,) float32 (when atom_features=True)
        charge       : (N_atoms,) float32 (when atom_features=True)
        is_backbone  : (N_atoms,) bool (when atom_features=True)
        hetero       : (N_atoms,) bool (when atom_features=True)
        atom_name    : list[str] of length N_atoms
        element      : list[str] of length N_atoms
        residue_index: (N_atoms,) int64, atom -> residue index map

    Always populated:
        chain_id         : list[str], length N_res
        residue_number   : (N_res,) int64
        insertion_code   : list[str], length N_res
        residue_name     : list[str], length N_res
        is_amino_acid    : (N_res,) bool

    Graph-level when ``energy=True``:
        proteon_energy_<component> : 0-dim float64 tensor for each component
                                     in proteon's energy dict
        proteon_ff                 : the force-field name string

    Args:
        atom_features: when False at atom granularity, skips per-atom
            ``atom_sasa``/``charge``/``is_backbone``/``hetero`` but still
            emits ``pos`` / ``residue_index`` / atom names. Ignored at
            residue granularity.
    """
    structure = proteon.load(str(pdb_path))

    feats = _compute_features_for_structure(
        structure,
        sasa=sasa,
        dssp=dssp,
        energy=energy,
        atom_sasa=(granularity == "atom" and atom_features),
        hbond_count=hbond_count,
        dihedrals=dihedrals,
        ff=ff,
        sasa_radii=sasa_radii,
    )

    return _data_from_structure(
        structure,
        feats,
        granularity=granularity,
        sasa=sasa,
        dssp=dssp,
        energy=energy,
        hbond_count=hbond_count,
        dihedrals=dihedrals,
        atom_features=atom_features,
        ff=ff,
    )


def proteon_pyg_data_batch(
    pdb_paths: Sequence[str | Path],
    *,
    granularity: Granularity = "residue",
    sasa: bool = True,
    dssp: bool = True,
    energy: bool = True,
    hbond_count: bool = False,
    dihedrals: bool = False,
    atom_features: bool = True,
    ff: str = "charmm19_eef1",
    sasa_radii: str = "bondi",
    n_threads: int | None = None,
) -> list["Data"]:
    """Build PyG ``Data`` objects for many PDBs using proteon's batch primitives.

    Per-structure result equals :func:`proteon_pyg_data`. Loads structures via
    ``proteon.batch_load`` and dispatches each enabled feature through its
    corresponding ``batch_*`` primitive (added in proteon 0.2.0). Strict mode
    only — one bad PDB raises.

    Args:
        pdb_paths: Sequence of paths to PDB files.
        granularity: ``"residue"`` or ``"atom"`` — applied to every Data in
            the batch.
        n_threads: Thread count for the proteon batch calls.
            ``None`` / ``-1`` / ``0`` = all cores.

    Returns:
        List of ``Data`` objects in input order. Use
        ``torch_geometric.data.Batch.from_data_list(out)`` to collate them.
    """
    paths = [str(p) for p in pdb_paths]
    if not paths:
        return []

    structures = proteon.batch_load(paths, n_threads=n_threads)

    feats_per: list[dict[str, Any]] = [{} for _ in structures]

    if sasa:
        residue_sasas = proteon.batch_residue_sasa(
            structures, radii=sasa_radii, n_threads=n_threads
        )
        rsas = proteon.batch_relative_sasa(
            structures, radii=sasa_radii, n_threads=n_threads
        )
        for i, (rs, rsa) in enumerate(zip(residue_sasas, rsas)):
            feats_per[i]["residue_sasa"] = rs
            feats_per[i]["rsa"] = rsa

    if dssp:
        codes = proteon.batch_dssp(structures, n_threads=n_threads)
        for i, code in enumerate(codes):
            feats_per[i]["dssp"] = code

    if energy:
        energies = proteon.batch_compute_energy(
            structures, ff=ff, n_threads=n_threads
        )
        for i, e in enumerate(energies):
            feats_per[i]["energy"] = e

    if granularity == "atom" and atom_features:
        per_atom = proteon.batch_atom_sasa(
            structures, radii=sasa_radii, n_threads=n_threads
        )
        for i, a in enumerate(per_atom):
            feats_per[i]["atom_sasa"] = a

    if hbond_count:
        counts = proteon.batch_hbond_count(structures, n_threads=n_threads)
        for i, c in enumerate(counts):
            feats_per[i]["hbond_count"] = c

    if dihedrals:
        triples = proteon.batch_dihedrals(structures, n_threads=n_threads)
        for i, (phi, psi, omega) in enumerate(triples):
            feats_per[i]["phi"] = phi
            feats_per[i]["psi"] = psi
            feats_per[i]["omega"] = omega

    return [
        _data_from_structure(
            s,
            f,
            granularity=granularity,
            sasa=sasa,
            dssp=dssp,
            energy=energy,
            hbond_count=hbond_count,
            dihedrals=dihedrals,
            atom_features=atom_features,
            ff=ff,
        )
        for s, f in zip(structures, feats_per)
    ]


class ProteonFeatures:
    """PyG ``BaseTransform`` that attaches proteon features to a ``Data``.

    Each input ``Data`` must carry a ``pdb_path`` attribute (str or Path)
    pointing to the PDB the Data was built from. The transform mutates the
    Data in place, attaching the same tensor schema as :func:`proteon_pyg_data`.

    Use as you would any PyG transform:

        >>> from proteon_pyg import ProteonFeatures
        >>> tf = ProteonFeatures(granularity="atom", hbond_count=True, dihedrals=True)
        >>> data = tf(data)  # data.pdb_path must be set
    """

    def __init__(
        self,
        *,
        granularity: Granularity = "residue",
        sasa: bool = True,
        dssp: bool = True,
        energy: bool = True,
        hbond_count: bool = False,
        dihedrals: bool = False,
        atom_features: bool = True,
        ff: str = "charmm19_eef1",
        sasa_radii: str = "bondi",
    ) -> None:
        self.granularity = granularity
        self.sasa = sasa
        self.dssp = dssp
        self.energy = energy
        self.hbond_count = hbond_count
        self.dihedrals = dihedrals
        self.atom_features = atom_features
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
            granularity=self.granularity,
            sasa=self.sasa,
            dssp=self.dssp,
            energy=self.energy,
            hbond_count=self.hbond_count,
            dihedrals=self.dihedrals,
            atom_features=self.atom_features,
            ff=self.ff,
            sasa_radii=self.sasa_radii,
        )
        for key, value in new.to_dict().items():
            setattr(data, key, value)
        return data

    def __repr__(self) -> str:
        flags = ", ".join(
            f"{k}={getattr(self, k)}"
            for k in ("granularity", "sasa", "dssp", "energy", "hbond_count", "dihedrals")
        )
        return f"ProteonFeatures({flags}, ff={self.ff!r})"
