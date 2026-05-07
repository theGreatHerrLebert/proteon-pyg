# Changelog

All notable changes to proteon-pyg are recorded here.

## [0.0.3] — atom-level granularity + batch helper

### Added (v0.0.3 — batch helper)

- `proteon_pyg.proteon_pyg_data_batch(pdb_paths, ...)` — load and
  feature-attach many PDBs in one parallel proteon call. Uses
  `proteon.batch_load` + `batch_residue_sasa` / `batch_relative_sasa` /
  `batch_atom_sasa` / `batch_dssp` / `batch_dihedrals` /
  `batch_hbond_count` / `batch_compute_energy` (all added in proteon
  0.2.0). Strict mode only — one bad PDB raises.
- EVIDENT batch-parity claim asserting batch output equals a Python
  loop of `proteon_pyg_data` at both residue and atom granularity, by
  full attribute-set equality.

### Added (v0.0.2 — atom-level granularity)

- `granularity="atom"` parameter on `proteon_pyg_data` and
  `ProteonFeatures`. Atom-level Data is dual-resolution: per-atom
  tensors (`pos`, `atom_sasa`, `charge`, `is_backbone`, `hetero`,
  `atom_name`, `element`) live alongside per-residue tensors at their
  natural shape, linked by `residue_index` (N_atoms,) → residue index.
  This matches PyG idioms: residue-level features stay (N_res,) instead
  of being broadcast to (N_atoms,).
- EVIDENT atom-level parity claim. Asserts every per-atom tensor value
  equals direct proteon API output (atom_sasa, charge, is_backbone,
  hetero, atom_name, element, residue_index) and that residue features
  remain parity-equal at residue resolution.

## [0.0.1] — initial scaffold

### Added

- `proteon_pyg.proteon_pyg_data(pdb_path, ...)` — build a residue-level
  PyG `Data` from a PDB with proteon features as tensor attributes.
- `proteon_pyg.ProteonFeatures` — PyG `BaseTransform` that attaches the
  same tensors to a `Data` carrying `data.pdb_path`.
- `proteon_pyg.encode_dssp` / `decode_dssp` / `DSSP_CLASSES` — integer
  round-trip for the 8-state DSSP alphabet.
- EVIDENT parity claim (`evident.yaml#proteon-pyg-residue-feature-parity`)
  asserting tensor values exact-equal to direct proteon API outputs.
- README documenting the tensor schema and the package's relationship
  to proteon-graphein.
