# proteon-pyg: EVIDENT Case Summary

Source repo: `.` (this directory)

## Problem

`proteon-pyg` attaches `proteon`-computed structural features (per-residue
SASA, relative SASA, 8-state DSSP, backbone H-bond counts, phi/psi/omega
dihedrals, and CHARMM19+EEF1 / AMBER96 total energy) to a PyTorch
Geometric `Data` object, skipping the `nx.Graph` round-trip that
[proteon-graphein](https://github.com/theGreatHerrLebert/proteon-graphein)
introduces.

PyG `Data` adapters are a quiet failure surface in geometric-DL
pipelines: numpy values go in, named tensors come out, and downstream
models have no way to tell whether the adapter preserved the source
values or silently rounded, reordered, or stripped NaN positions during
the numpy → torch transcription. Every model trained on this adapter's
output inherits any bias it introduces.

## Trust Strategy

Validation, with `proteon` itself as the oracle.

The adapter does not *compute* features. It calls `proteon` and then
transcribes those numpy outputs into PyG-native tensor attributes,
honoring the residue-iteration order of `structure.residues`. The trust
question is therefore narrow: **do the tensor values that land on the
`Data` object equal the values `proteon` returned for the same residue,
including NaN positions and integer-coded DSSP states?**

Because the oracle and the implementation share a process and a
`proteon` version, the tolerance is exact equality (post `.numpy()`
round-trip). Any drift means the adapter mutated a value during
transcription, not that `proteon` is non-deterministic.

## Evidence

- `tests/test_parity.py` — six parity tests covering:
  - per-residue SASA / RSA / DSSP / hbond_count / phi/psi/omega equality
    (NaN-aware)
  - graph-level energy dict mapped to per-component scalar tensors
  - HETATM residues never carry a DSSP int code other than `-1`
  - atom-level: per-atom pos / atom_sasa / charge / is_backbone / hetero
    / atom_name / element / residue_index equal direct proteon API
    output, with residue-level tensors still satisfying the residue
    parity claim
  - `proteon_pyg_data_batch(paths)` produces Data objects with exactly
    the same attribute-set and tensor values as a Python loop calling
    `proteon_pyg_data(p)` per path — at both residue and atom granularity
- Fixtures: `1crn.pdb` (pure protein) and `1ake.pdb` (protein + waters/
  ligand). Tests skip cleanly when the fixtures are absent.
- DSSP encoding round-trip (`tests/test_features.py`) catches the case
  where proteon's `dssp()` output drifts away from the 8-state alphabet.

## Assumptions

- The adapter runs in the same process as the oracle; `proteon` version
  skew is not in scope for this manifest.
- `torch` and `torch_geometric` are installed when the parity test runs.
  When missing, the test skips via `pytest.importorskip` rather than
  passing silently — the optional `[pyg]` extra installs both.
- `DSSP_CLASSES = ('H', 'B', 'E', 'G', 'I', 'T', 'S', 'C')` is the
  canonical alphabet emitted by `proteon.dssp()`. Any character outside
  this set encodes to `-1` and decodes to `?`.
- Test fixtures live at `/scratch/TMAlign/proteon/test-pdbs/` and tests
  skip when those files are missing.

## Failure Modes

- proteon's `dssp()` output grows a 9th state symbol that
  `encode_dssp`/`decode_dssp` does not know about — caught by the
  round-trip test and the publicly exported `DSSP_CLASSES` constant
  letting users detect the drift.
- The order of residues iterated by `structure.residues` changes between
  proteon versions, decoupling tensor indices from the per-AA arrays
  returned by `dssp()` / `hbond_count()` / `backbone_dihedrals()` —
  caught because the parity test rebuilds the AA index from
  `data.is_amino_acid` before comparing.
- NaN is silently coerced to 0.0 (or a real number) inside torch's
  `from_numpy` path, breaking the NaN-position parity at chain termini —
  caught by explicit `math.isnan` checks at every dihedral index in
  the parity test.

## What Is Still Lacking (Deferred Claims)

The atom-level and batch parity claims that were deferred at v0.0.1 are
both promoted into the manifest as tier-`ci` claims as of v0.0.3, with
real oracle, tolerance, command, and artifact entries. `deferred_claims:`
is currently empty.

Future work that may grow into manifest claims:
- Tolerant batch loading (skip-on-fail mode) needs its own claim shape
  because the input → output index correspondence stops being identity.
- Accepting pre-loaded `proteon.Structure` objects directly is a
  convenience refactor with no new trust property to claim.

## EVIDENT Lessons

- A tensor adapter has the same claim shape as a graph adapter: the
  trust question is about transcription, not computation. Reuse the
  parity-against-oracle pattern from `proteon-graphein` rather than
  inventing new tolerances.
- NaN handling is a real claim. PyG `Data` attributes are tensors, not
  attribute dicts, so "skip the attribute when proteon returns NaN" is
  not an option — the tensor index has to carry NaN itself, and the
  parity test has to assert that explicitly.
- Sentinel values for non-AA residues (`-1` for DSSP and hbond_count)
  are part of the public contract, not implementation noise. They
  belong in the manifest's claim text.
