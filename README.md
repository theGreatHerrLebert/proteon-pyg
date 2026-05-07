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

Pre-release (v0.0.1). Single-PDB, residue-level only — every node
corresponds to one proteon residue (AA + non-AA). DSSP is integer-encoded
(0..7 per `DSSP_CLASSES`) with `-1` for non-AA residues; a `decode_dssp`
helper reverses the encoding. Atom-level granularity and a batch helper
follow in v0.0.2 and v0.0.3.

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

Build a `Data` straight from a PDB:

```python
from proteon_pyg import proteon_pyg_data

data = proteon_pyg_data(
    "1crn.pdb",
    sasa=True, dssp=True, energy=True,
    hbond_count=True, dihedrals=True,
)

print(data.residue_sasa.shape)        # (N,)
print(data.dssp.shape)                # (N,) int64, 0..7 per DSSP_CLASSES
print(data.proteon_energy_total)      # 0-dim tensor, kJ/mol
print(data.proteon_ff)                # 'charmm19_eef1'
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

## Tensor schema (residue-level)

| Attribute | Shape | dtype | Notes |
|-----------|-------|-------|-------|
| `pos` | (N, 3) | float32 | CA coords; first-atom fallback for residues without a CA; NaN if a residue has no atoms |
| `chain_id` | list[str] | — | length N |
| `residue_number` | (N,) | int64 | |
| `insertion_code` | list[str] | — | empty string when absent |
| `residue_name` | list[str] | — | 3-letter codes, length N |
| `is_amino_acid` | (N,) | bool | |
| `residue_sasa` | (N,) | float32 | Å² |
| `rsa` | (N,) | float32 | NaN for non-standard residues |
| `dssp` | (N,) | int64 | 0..7 per `DSSP_CLASSES`, -1 for non-AA |
| `hbond_count` | (N,) | int64 | -1 for non-AA |
| `phi`, `psi`, `omega` | (N,) | float32 | NaN at chain termini and non-AA residues |
| `proteon_energy_<comp>` | () | float64 | one 0-dim tensor per energy component |
| `proteon_ff` | str | — | force-field name |

## Why a separate package

proteon-graphein already attaches the same values to a Graphein graph.
proteon-pyg exists for users who want PyG without paying for Graphein's
dependency tree, and to avoid the `nx.Graph` → `Data` round-trip in
training pipelines. Both packages share proteon as their oracle and target
exact equality against direct proteon API calls.

## Roadmap

- v0.0.1 (current): residue-level `Data`, single PDB, `proteon_pyg_data`
  + `ProteonFeatures` transform, parity claim against proteon.
- v0.0.2: atom-level granularity (`pos` per atom, plus `atom_sasa`,
  `charge`, `is_backbone`, `hetero`, residue indexing).
- v0.0.3: batch helper using proteon's batch primitives + a
  batch-equals-loop parity claim.

## Trust

This package follows the
[EVIDENT](https://github.com/theGreatHerrLebert/evident) claim-based
evidence workflow. Its current trust claim is documented in
[`evident.yaml`](evident.yaml) and explained in [`CASE.md`](CASE.md):

- **Parity claim** (tier `ci`): every value attached to the `Data` object
  is exactly equal (post `.numpy()` round-trip) to what proteon returned
  for that residue. Reproduced by `pytest tests/test_parity.py`.

## License

MIT — see [LICENSE](LICENSE).
