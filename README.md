# proteon-pyg

[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) integration
for [proteon](https://github.com/theGreatHerrLebert/proteon) — exposes
per-residue SASA, relative SASA, 8-state DSSP, backbone H-bond counts,
phi/psi/omega dihedrals, and CHARMM19+EEF1 / AMBER96 force-field energies
as tensor attributes on a PyG `Data` object.

Sibling project of [proteon-graphein](https://github.com/theGreatHerrLebert/proteon-graphein),
sharing the same parity-against-proteon trust model. proteon-graphein
attaches features to a Graphein `nx.Graph`; proteon-pyg attaches the same
values directly to a PyG `Data` and skips the Graphein hop.

## Status

Pre-release (v0.0.3). Residue and atom-level granularity, single-PDB and
batch helpers, DSSP int-encoded with a `decode_dssp` round-trip. Atom-level
Data is dual-resolution — per-atom tensors live alongside per-residue
tensors, linked via `data.residue_index`. Parity-tested against proteon
direct API on `1crn.pdb` (residue + atom) and `1ake.pdb` (HETATM), plus
batch-equals-loop parity at both granularities.

## Install (development)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[pyg,dev]
```

The `pyg` extra brings in `torch` and `torch_geometric`. They are not
core dependencies — proteon-pyg is importable without them, and the test
suite skips the PyG-dependent tests when they are missing.

## Usage

Residue-level Data (one node per residue):

```python
from proteon_pyg import proteon_pyg_data

data = proteon_pyg_data(
    "1crn.pdb",
    sasa=True, dssp=True, energy=True,
    hbond_count=True, dihedrals=True,
)
print(data.residue_sasa.shape)        # (N_res,)
print(data.dssp.shape)                # (N_res,) int64, 0..7 per DSSP_CLASSES
print(data.proteon_energy_total)      # 0-dim tensor, kJ/mol
```

Atom-level Data (dual-resolution — per-atom + per-residue tensors):

```python
data = proteon_pyg_data("1crn.pdb", granularity="atom", hbond_count=True)
print(data.pos.shape)                 # (N_atoms, 3) atom coords
print(data.atom_sasa.shape)           # (N_atoms,)
print(data.residue_sasa.shape)        # (N_res,)  — natural residue shape
print(data.residue_index.shape)       # (N_atoms,) atom -> residue index
# Broadcast residue features to atoms when needed:
per_atom_dssp = data.dssp[data.residue_index]
```

Batch over many PDBs in one parallel proteon call:

```python
from proteon_pyg import proteon_pyg_data_batch

datas = proteon_pyg_data_batch(
    ["1crn.pdb", "1ake.pdb", "1ubq.pdb"],
    granularity="atom",
    hbond_count=True,
    n_threads=-1,
)

# Collate via PyG's standard batching
from torch_geometric.data import Batch
batch = Batch.from_data_list(datas)
```

Slot into a PyG `Dataset` pipeline:

```python
from torch_geometric.transforms import Compose
from proteon_pyg import ProteonFeatures

transform = ProteonFeatures(hbond_count=True, dihedrals=True)

# Each Data passed to transform must carry data.pdb_path:
class MyDataset(InMemoryDataset):
    def process(self):
        data_list = []
        for path in self.pdb_paths:
            d = Data()
            d.pdb_path = path
            data_list.append(d)
        ...
        if self.transform is not None:
            data_list = [self.transform(d) for d in data_list]
        ...

ds = MyDataset(transform=transform)
```

Round-tripping DSSP between integer codes and the canonical 8-character
string:

```python
from proteon_pyg import decode_dssp, encode_dssp, DSSP_CLASSES

assert decode_dssp(encode_dssp("HHHHEEC")) == "HHHHEEC"
print(DSSP_CLASSES)  # ('H', 'B', 'E', 'G', 'I', 'T', 'S', 'C')
```

## Tensor schema

**Residue-level** (`granularity="residue"`, default — `N = N_res`):

| Attribute | Shape | dtype | Notes |
|-----------|-------|-------|-------|
| `pos` | (N, 3) | float32 | CA coords; first-atom fallback; NaN if no atoms |
| `chain_id` | list[str] | — | length N |
| `residue_number` | (N,) | int64 | |
| `insertion_code` | list[str] | — | empty string when absent |
| `residue_name` | list[str] | — | 3-letter codes |
| `is_amino_acid` | (N,) | bool | |
| `residue_sasa` | (N,) | float32 | Å² |
| `rsa` | (N,) | float32 | NaN for non-standard residues |
| `dssp` | (N,) | int64 | 0..7 per `DSSP_CLASSES`, -1 for non-AA |
| `hbond_count` | (N,) | int64 | -1 for non-AA |
| `phi`, `psi`, `omega` | (N,) | float32 | NaN at chain termini and non-AA |
| `proteon_energy_<comp>` | () | float64 | one 0-dim tensor per energy component |
| `proteon_ff` | str | — | force-field name |

**Atom-level** (`granularity="atom"`) carries the same residue tensors at
residue resolution **plus** these per-atom tensors:

| Attribute | Shape | dtype | Notes |
|-----------|-------|-------|-------|
| `pos` | (N_atoms, 3) | float32 | atom coordinates (replaces residue `pos`) |
| `residue_index` | (N_atoms,) | int64 | atom → residue index in the same Data |
| `atom_name` | list[str] | — | length N_atoms |
| `element` | list[str] | — | length N_atoms |
| `atom_sasa` | (N_atoms,) | float32 | per-atom SASA (Å²) |
| `charge` | (N_atoms,) | float32 | partial charge |
| `is_backbone` | (N_atoms,) | bool | |
| `hetero` | (N_atoms,) | bool | HETATM flag |

## Why a separate package

proteon-graphein already attaches the same values to a Graphein graph.
proteon-pyg exists for users who want PyG without paying for Graphein's
dependency tree, and to avoid the `nx.Graph` → `Data` round-trip in
training pipelines. Both packages share proteon as their oracle and target
exact equality against direct proteon API calls.

## Roadmap

- v0.0.1: residue-level `Data`, single PDB, `proteon_pyg_data` +
  `ProteonFeatures` transform, parity claim against proteon.
- v0.0.2: atom-level granularity — dual-resolution Data (per-atom
  tensors alongside per-residue tensors, linked via `residue_index`).
- v0.0.3 (current): `proteon_pyg_data_batch` using proteon's
  rayon-parallel batch primitives. Strict mode only. Batch-equals-loop
  parity claim at both granularities.
- v0.0.x follow-ups: tolerant batch loading; accepting pre-loaded
  `proteon.Structure` objects directly; PyPI release.

## Trust

This package follows the
[EVIDENT](https://github.com/theGreatHerrLebert/evident) claim-based
evidence workflow. Its current trust claim is documented in
[`evident.yaml`](evident.yaml) and explained in [`CASE.md`](CASE.md):

- **Residue-feature parity** (tier `ci`): every residue-level tensor
  is exactly equal (post `.numpy()` round-trip) to direct proteon output.
- **Atom-feature parity** (tier `ci`): every per-atom tensor (pos,
  atom_sasa, charge, is_backbone, hetero, atom_name, element,
  residue_index) equals direct proteon API output, with residue-level
  tensors continuing to satisfy the residue-feature claim.
- **Batch parity** (tier `ci`): `proteon_pyg_data_batch(paths)[i]`
  produces a Data with exactly the same attribute set and tensor values
  as `proteon_pyg_data(paths[i])`, at both granularities.

All three are reproduced by `pytest tests/test_parity.py`.

## License

MIT — see [LICENSE](LICENSE).
