# Changelog

All notable changes to proteon-pyg are recorded here.

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
