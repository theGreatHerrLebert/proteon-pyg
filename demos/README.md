# proteon-pyg demos

Two scripts that exercise proteon-pyg end-to-end on real PDBs and emit
artifacts (plots, console summaries) you can eyeball.

These are **sanity** and **infrastructure** demos, not science demos.
They prove:

- The adapter's tensors land in physically plausible ranges.
- The NaN-position contract (terminal dihedrals, non-AA RSA) holds on
  real structures.
- The batch helper actually parallelizes — running through the adapter
  does not destroy proteon's rayon parallelism.

They do **not** prove:

- That proteon features improve any downstream model. That's a research
  claim with its own pre-declared tolerance and decision rule, and
  belongs in a separate notebook with its own EVIDENT manifest. See
  the deferred-claim note in [`../CASE.md`](../CASE.md).

## Running

```bash
# From the repo root, with [pyg] dependencies installed:
/scratch/TMAlign/proteon/.venv/bin/python demos/sanity_demo.py
/scratch/TMAlign/proteon/.venv/bin/python demos/infra_demo.py --n 100
```

Output PNGs land under `demos/output/{sanity,infra}/`.

## sanity_demo.py

Loads 5 small PDBs (`1crn`, `1aaj`, `1ake`, `1bpi`, `1ubq`) via
`proteon_pyg_data_batch`, then:

1. Emits a Ramachandran scatter from `data.phi` × `data.psi`.
2. Histograms `data.residue_sasa` and the NaN-filtered `data.rsa`.
3. Bar-plots DSSP-8 class frequencies.
4. Compares CHARMM19+EEF1 energy components across the 5 PDBs.

Then runs range checks on every Data:

- `residue_sasa >= 0`
- `rsa >= 0` at AA positions (NaN allowed at non-standard residues)
- `phi`, `psi`, `omega` in `[-180, 180]` where defined
- `dssp` in `{-1, 0..7}`
- exactly 1 NaN at chain termini for phi and psi on `1crn` (single-chain)

Exits 0 with `[sanity] PASS` when every check holds.

## infra_demo.py

Times the proteon-pyg integration against a Python loop, at several
thread counts, and plots the speedup curve. Uses the 10K-PDB validation
corpus at `/scratch/TMAlign/proteon/validation/pdbs_10k/`.

```text
loop          : single-call proteon_pyg_data per PDB, in a Python loop
batch n=1     : proteon_pyg_data_batch with one rayon thread
batch n=4..   : proteon_pyg_data_batch with progressively more threads
```

### Observed numbers (16-core machine, 70 small PDBs)

```text
[infra]   loop:             4.76s  ( 14.7 PDBs/s)
[infra]   batch n= 1:      25.23s  (  2.8 PDBs/s, speedup x0.19)
[infra]   batch n= 4:       8.80s  (  8.0 PDBs/s, speedup x0.54)
[infra]   batch n= 8:       7.53s  (  9.3 PDBs/s, speedup x0.63)
[infra]   batch n=16:       7.26s  (  9.6 PDBs/s, speedup x0.66)
```

### Honest findings

The demo surfaces two things parity tests can't:

1. **86% of the 10K validation corpus fails proteon's loader** with a
   `LooseWarning: SEQRES residue total invalid` error from pdbtbx.
   This is corpus-specific (the validation corpus is intentionally a
   stress-test mix) and is upstream of proteon-pyg, but worth being
   aware of when you point a real dataset at the adapter.

2. **The batch helper is currently slower than the Python loop on
   small PDBs**, peaking at ~0.66× the loop's wall time at full
   parallelism. Decomposed:
   - `proteon.batch_load` is ~2.5× faster than a `[proteon.load(p) for p in paths]` loop.
   - `proteon.batch_residue_sasa` is ~0.54× the speed of a per-structure
     loop because each PyRef extraction serializes through the GIL on the
     main thread before the rayon pool gets to do real work, and the
     per-structure compute on small PDBs is too short to amortize that
     cost.

   This is an upstream perf characteristic of the batch primitives
   added in [proteon PR #66](https://github.com/theGreatHerrLebert/proteon/pull/66),
   not a proteon-pyg bug. Worth filing a follow-up issue upstream:
   for small-structure batches, the Python overhead of marshaling
   `Vec<PyRef<PyPDB>>` dominates the gain from rayon. Likely fix is to
   pre-extract all PDB views inside one GIL release and pass plain
   owned data into the rayon body.

   Practical takeaway: today the batch helper is a correctness
   convenience, not a performance win for typical-sized PDBs. Use it
   when you want the same code path as a future fast version, or for
   `batch_load` alone via the public proteon API.
